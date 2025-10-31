import os
import struct
import h5py
import lmdb
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.special import softmax
import matplotlib.pyplot as plt
from umap import UMAP

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from SimCLR_args import conf
from SimCLR import Network


def compute_latent_vectors(args, dataset_loader, device, model):
	model.eval()
	latent_list = []
	with torch.no_grad():
		for batch in tqdm(dataset_loader, desc="Computing latent vectors"):
			imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
			imgs = imgs.to(device)
			with torch.cuda.amp.autocast():
				lat = model(imgs)
			lat = lat.detach().cpu().view(lat.size(0), -1).numpy()
			latent_list.append(lat)
	return np.vstack(latent_list)


def create_labeled_lmdb(src_lmdb, dst_lmdb, labels, map_size=None, safety_factor=1.2, min_map_size=(15 << 30)):
	# Labels is a 1D array aligned with keys/order in src_lmdb
	with lmdb.open(src_lmdb, readonly=True, lock=False, readahead=False) as env:
		with env.begin() as txn:
			keys = [key for key, _ in txn.cursor()]

	assert len(keys) == len(labels), "Number of labels must match number of entries in source LMDB"

	# estimating by summing sizes of values in source LMDB and add 4 bytes per entry for label
	if map_size is None:
		total_bytes = 0
		with lmdb.open(src_lmdb, readonly=True, lock=False, readahead=False) as env:
			with env.begin() as txn:
				cursor = txn.cursor()
				for _, value in tqdm(cursor, desc="Estimating LMDB size"):
					total_bytes += len(value)
		total_bytes += len(keys) * 4
		estimated = int(total_bytes * float(safety_factor))
		map_size = max(estimated, int(min_map_size))
		print(f"Estimated map_size={map_size} bytes (safety_factor={safety_factor})")

	# Write to new LMDB. Value format: 4 bytes unsigned int label (little-endian) + image bytes
	os.makedirs(os.path.dirname(dst_lmdb), exist_ok=True)
	env_out = lmdb.open(dst_lmdb, map_size=map_size)
	with env_out.begin(write=True) as out_txn, lmdb.open(src_lmdb, readonly=True, lock=False, readahead=False).begin() as in_txn:
		cursor = in_txn.cursor()
		for idx, (key, value) in enumerate(tqdm(cursor, desc="Writing labeled LMDB", total=len(keys))):
			label = int(labels[idx])
			packed_label = struct.pack('<I', label)
			out_txn.put(key, packed_label + value)
	env_out.sync()
	env_out.close()


def main(num_clusters=0, top_k=0):
	args = conf()
	# device
	use_cuda = not args.no_cuda and torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	cudnn.deterministic = True
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	# Dataset transform (single deterministic view)
	_transform = transforms.Compose([
		transforms.Resize((args.r_crop_size, args.r_crop_size)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406],
							 std=[0.229, 0.224, 0.225])
	])

	# Dataset loader (LMDB or folder)
	if args.lmdb:
		from DataLoaderLMDB import Dataset_
		dataset = Dataset_(args.path2db, transform=_transform, train_portion=1.0, shuffle=False, val=True, seed=args.seed)
	else:
		from torchvision.datasets import ImageFolder
		dataset = ImageFolder(args.path2db, transform=_transform)

	loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

	# Load model
	model = Network(args)
	if args.resume and os.path.isfile(args.resume):
		print(f"Loading checkpoint {args.resume}")
		checkpoint = torch.load(args.resume, map_location=device)
		state_dict = checkpoint.get('state_dict', checkpoint)
		new_state = {k.replace('module.', ''): v for k, v in state_dict.items()}
		model.load_state_dict(new_state)
	else:
		raise FileNotFoundError(f"No checkpoint found at {args.resume}")

	if not args.no_multigpu and torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
	model = model.to(device)

	# 1) compute latent vectors
	latent_vectors = compute_latent_vectors(args, loader, device, model)
	latent_vectors /= np.linalg.norm(latent_vectors, axis=1, keepdims=True)

	# 2) k-means clustering
	num_samples = latent_vectors.shape[0]
	if num_clusters == 0:
		candidates = [64, 128, 256, 512, 1024, 2048, 4096]
		candidates = [c for c in candidates if 2 <= c < num_samples]
		if not candidates:
			candidates = [max(2, min(256, max(2, num_samples // 10)))]

		sample_size = min(10000, num_samples)
		sample_idx = np.random.choice(num_samples, sample_size, replace=False)
		latent_sample = latent_vectors[sample_idx]

		best_k = None
		best_score = -1.0
		print("Auto-selecting number of clusters using silhouette score over candidates:", candidates)
		for c in candidates:
			try:
				kmeans_c = KMeans(n_clusters=c, random_state=args.seed, n_init=10)
				labels_c = kmeans_c.fit_predict(latent_sample)
				if len(set(labels_c)) <= 1:
					continue
				score = silhouette_score(latent_sample, labels_c)
				print(f" k={c} silhouette={score:.4f}")
				if score > best_score:
					best_score = score
					best_k = c
			except Exception as e:
				print(f" k={c} failed: {e}")

		num_clusters = best_k if best_k is not None else candidates[0]
		print(f"Selected num_clusters={num_clusters}")

	print(f"Running KMeans with {num_clusters} clusters on {num_samples} samples")
	kmeans = KMeans(n_clusters=num_clusters, random_state=args.seed, n_init="auto")
	cluster_labels = kmeans.fit_predict(latent_vectors)
	cluster_centers = kmeans.cluster_centers_

	# 3) Save pseudo-labels h5
	os.makedirs('data', exist_ok=True)
	pseudo_h5 = os.path.join('data', 'pseudo_labels_ldl.h5')
	print(f"Saving pseudo-labels to {pseudo_h5}")
	with h5py.File(pseudo_h5, 'w') as f:
		f.create_dataset('hard_labels', data=cluster_labels.astype(np.int32))
		f.create_dataset('cluster_centers', data=cluster_centers)

	# 4) Generate UMAP visualization
	print("Generating UMAP visualization...")
	subset_for_umap = 100000
	if num_samples > subset_for_umap:
		idx = np.random.choice(num_samples, subset_for_umap, replace=False)
		latent_subset = latent_vectors[idx]
		cluster_subset = cluster_labels[idx]
	else:
		latent_subset = latent_vectors
		cluster_subset = cluster_labels

	reducer = UMAP(n_components=2, random_state=args.seed, n_neighbors=15, min_dist=0.1, metric="euclidean")
	latent_2d = reducer.fit_transform(latent_subset)
	np.save(os.path.join('data', 'udp_map.npy'), latent_2d)

	plt.figure(figsize=(10, 8))
	plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=cluster_subset, cmap="tab20", s=2, alpha=0.7)
	plt.title(f"UMAP Projection - {num_clusters} clusters")
	plt.xlabel("UMAP-1")
	plt.ylabel("UMAP-2")
	plt.tight_layout()
	plt.savefig(os.path.join('data', 'udp_map.png'), dpi=300)
	plt.close()
	print("Saved UMAP visualization to data/udp_map.png")

	# 5) Create labeled LMDB (works for both LMDB and folder input)
	src = args.path2db
	dst = './data/imagenet1k/imagenet1k_train_labelled.lmdb'
	dst = os.path.abspath(dst)

	if args.lmdb:
		print(f"Creating labeled LMDB at {dst} (source is LMDB)")
		create_labeled_lmdb(src, dst, cluster_labels)
		print("Done. New LMDB created with cluster labels embedded in values.")
	else:
		print("Dataset is in folder format — building labeled LMDB from image files.")
		import io
		image_paths = [s[0] for s in dataset.samples]
		assert len(image_paths) == len(cluster_labels), "Mismatch between images and cluster labels"

		total_bytes = 0
		for img_path in tqdm(image_paths, desc="Estimating total image size"):
			total_bytes += os.path.getsize(img_path)
		map_size = int(total_bytes * 1.2)
		os.makedirs(os.path.dirname(dst), exist_ok=True)
		print(f"Estimated LMDB map size = {map_size / (1<<30):.2f} GB")

		env = lmdb.open(dst, map_size=map_size)
		with env.begin(write=True) as txn:
			for idx, (img_path, label) in enumerate(tqdm(zip(image_paths, cluster_labels),
														desc="Writing labeled LMDB",
														total=len(image_paths))):
				with open(img_path, 'rb') as f:
					img_bytes = f.read()
				key = f"{idx:08d}".encode('ascii')
				packed_label = struct.pack('<I', int(label))
				value = packed_label + img_bytes
				txn.put(key, value)
		env.sync()
		env.close()
		print(f"Labeled LMDB successfully created at: {dst}")


if __name__ == '__main__':
	main(num_clusters=0, top_k=0)
