import random
from tqdm import tqdm
import h5py

import torch
import torch.backends.cudnn as cudnn

from SimCLR_args import conf
from SimCLR import *

import numpy as np
from sklearn.cluster import KMeans
from scipy.special import softmax


# Option
args = conf()
print(args)

if args.lmdb:
    from DataLoaderLMDB import Dataset_
else:
    from DataLoader import Dataset_

# GPUs
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# to deterministic
cudnn.deterministic = True
random.seed(args.seed)
torch.manual_seed(args.seed)


# _transform = transforms.Compose([
#     # transforms.RandomResizedCrop(size=args.r_crop_size),
#     # transforms.RandomHorizontalFlip(),
#     # transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
#     # transforms.RandomGrayscale(p=args.p_grayscale),
#     # RandomGaussianBlur(probability=args.p_blur, radius=torch.rand(1).item()*(args.max_blur_r-args.min_blur_r)+args.min_blur_r), #Random Gaussian blur with radius between 2 and 4
#     transforms.Resize((224, 224)),
#     transforms.ToTensor()
# ])

# #Loading dataset
# _portion = 1

# _dataset = Dataset_(args.path2traindb, transform=_transform, train_portion=_portion, shuffle=True ,val=False, seed=args.seed)
# _loader = DataLoader(_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)


# SimCLR = Network(args).to(device)
# checkpoint = torch.load('data/weights/SimCLR.tar', map_location=device)
# state_dict = checkpoint['state_dict']
# new_state_dict = {}
# for k, v in state_dict.items():
#     new_key = k.replace('module.', '')
#     new_state_dict[new_key] = v

# SimCLR.load_state_dict(new_state_dict)

# #obtaining latent vectors
# with h5py.File('latent_vectors.h5', 'a') as f:
#     first_batch = True
    
#     for [imgs, img_classes] in tqdm(iterable=_loader, desc="Computing latent vectors"):
#         imgs = imgs.to(device)
        
#         with torch.cuda.amp.autocast():
#             # Forward pass through SimCLR to obtain latent vectors
#             latent_vecs_batch = SimCLR(imgs).cpu().detach()
#             latent_vecs_batch = latent_vecs_batch.view(latent_vecs_batch.size(0), -1).numpy()
#             img_classes_np = np.array(img_classes)
        
#         # Initialize datasets if it's the first batch
#         if first_batch:
#             # Create or open the latent_vectors dataset
#             latent_vectors_dataset = f.create_dataset(
#                 'latent_vectors',
#                 shape=(0, latent_vecs_batch.shape[1]),
#                 maxshape=(None, latent_vecs_batch.shape[1]),
#                 chunks=True
#             )
            
#             # Create or open the classes dataset
#             classes_dataset = f.create_dataset(
#                 'classes',
#                 shape=(0,),
#                 maxshape=(None,),
#                 chunks=True
#             )
            
#             first_batch = False
#         else:
#             # Re-access the datasets in subsequent iterations
#             latent_vectors_dataset = f['latent_vectors']
#             classes_dataset = f['classes']

#         # Resize datasets to accommodate the new batch of data
#         latent_vectors_dataset.resize(latent_vectors_dataset.shape[0] + latent_vecs_batch.shape[0], axis=0)
#         classes_dataset.resize(classes_dataset.shape[0] + img_classes_np.shape[0], axis=0)

#         # Append the new batch of data to the resized datasets
#         latent_vectors_dataset[-latent_vecs_batch.shape[0]:] = latent_vecs_batch
#         classes_dataset[-img_classes_np.shape[0]:] = img_classes_np

# import faiss
# import h5py
# import numpy as np
# from tqdm import tqdm

# # Parameters
# K = 10  # Number of nearest neighbors
# latent_vector_dim = args.out_dim  # Dimensionality of latent vectors (from SimCLR)
# chunk_size = 10000  # Load latent vectors in chunks

# # Enable GPU resources for FAISS
# gpu_resources = faiss.StandardGpuResources()

# # Open the HDF5 file containing latent vectors and class labels
# with h5py.File('latent_vectors.h5', 'r') as f:
#     latent_vectors_dataset = f['latent_vectors']
#     classes_dataset = f['classes']

#     num_samples = latent_vectors_dataset.shape[0]
#     num_classes = int(np.max(classes_dataset) - np.min(classes_dataset)) + 1

#     # Create a FAISS index with product quantization on GPU
#     quantizer = faiss.IndexFlatL2(latent_vector_dim)  # Use L2 (Euclidean) distance
#     index_cpu = faiss.IndexIVFPQ(quantizer, latent_vector_dim, 100, 8, 8)  # PQ codes for compression
#     index_cpu.nprobe = 10  # Number of cells to search for better recall

#     # Transfer index to GPU
#     index = faiss.index_cpu_to_gpu(gpu_resources, 0, index_cpu)

#     # Train the index on a subset of the data (on GPU)
#     latent_chunk = latent_vectors_dataset[:4000].astype('float32')  # Use 4000 points for training
#     faiss.normalize_L2(latent_chunk)  # Normalize for cosine similarity
#     index.train(latent_chunk)  # Train the index

#     # Add vectors to the index in chunks (on GPU)
#     for chunk_start in tqdm(range(0, num_samples, chunk_size), desc="Building index"):
#         chunk_end = min(chunk_start + chunk_size, num_samples)

#         # Load a chunk of latent vectors from the dataset
#         latent_chunk = latent_vectors_dataset[chunk_start:chunk_end].astype('float32')

#         # Normalize the latent vectors for cosine similarity
#         faiss.normalize_L2(latent_chunk)

#         # Add the chunk of latent vectors to the FAISS index (on GPU)
#         index.add(latent_chunk)

#     # Save the index to disk
#     faiss.write_index(faiss.index_gpu_to_cpu(index), "faiss_index_ivfpq.index")

# # Reopen the HDF5 file containing latent vectors and class labels
# with h5py.File('latent_vectors.h5', 'r') as f:
#     latent_vectors_dataset = f['latent_vectors']
#     classes_dataset = f['classes']

#     num_samples = latent_vectors_dataset.shape[0]
#     num_classes = int(np.max(classes_dataset) - np.min(classes_dataset)) + 1

#     # Load the previously saved FAISS index and transfer it to GPU
#     index_cpu = faiss.read_index("faiss_index_ivfpq.index")
#     index = faiss.index_cpu_to_gpu(gpu_resources, 0, index_cpu)

#     # Initialize pseudo-labels array
#     pseudo_labels = np.zeros((num_samples, num_classes))

#     # Perform nearest neighbor search in chunks (on GPU)
#     for chunk_start in tqdm(range(0, num_samples, chunk_size), desc="Searching neighbors"):
#         chunk_end = min(chunk_start + chunk_size, num_samples)

#         # Load the current chunk of latent vectors
#         latent_chunk = latent_vectors_dataset[chunk_start:chunk_end].astype('float32')

#         # Normalize the latent vectors (important for cosine similarity)
#         faiss.normalize_L2(latent_chunk)

#         # Perform K-nearest neighbors search (on GPU)
#         D, I = index.search(latent_chunk, K)  # I contains indices of nearest neighbors

#         # Assign pseudo-labels based on the nearest neighbors
#         for i, knn_indices in enumerate(I):
#             # Sort indices to comply with h5py requirements
#             sorted_knn_indices = np.sort(knn_indices)
            
#             # Retrieve the class labels for the sorted nearest neighbors
#             knn_classes_sorted = classes_dataset[sorted_knn_indices]

#             # Reorder the class labels back to the original order of knn_indices
#             knn_classes = knn_classes_sorted[np.argsort(np.argsort(knn_indices))]

#             # Increment the pseudo-label count for the corresponding classes
#             for c in knn_classes:
#                 pseudo_labels[chunk_start + i, int(c) - int(np.min(classes_dataset))] += 1

#             # Normalize the pseudo-label distribution
#             pseudo_labels[chunk_start + i] /= np.sum(pseudo_labels[chunk_start + i])

# # Save the pseudo-labels
# np.save('data/pseudo_labels.npy', pseudo_labels)


# Parameters
num_clusters = 250   # Number of clusters for K-means
top_k = 5            # Number of closest clusters to use for each sample

# Load latent vectors into RAM
with h5py.File('latent_vectors.h5', 'r') as f:
    latent_vectors = np.array(f['latent_vectors']).astype('float32')  # Load latent vectors

# Normalize latent vectors
print("Normalizing latent vectors...")
latent_vectors /= np.linalg.norm(latent_vectors, axis=1, keepdims=True)

# Step 1: Apply K-means clustering
print("Applying K-means clustering...")
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(latent_vectors)
cluster_centers = kmeans.cluster_centers_

# Initialize the pseudo-labels array
print("Initializing the pseudo-labels array")
num_samples = latent_vectors.shape[0]
pseudo_labels = np.zeros((num_samples, num_clusters))

# Step 2: Calculate distances of each sample to each cluster center
# This step is memory-intensive, so we loop in batches if needed
batch_size = 10000  # Adjust based on available memory
for i in tqdm(range(0, num_samples, batch_size), desc="Assigning pseudo-labels..."):
    end = min(i + batch_size, num_samples)
    # Calculate pairwise distances for the batch
    distances = np.linalg.norm(latent_vectors[i:end, None] - cluster_centers, axis=2)
    
    # Step 3: Find the `top_k` closest clusters for each sample in the batch
    top_k_indices = np.argpartition(distances, top_k, axis=1)[:, :top_k]
    top_k_distances = np.take_along_axis(distances, top_k_indices, axis=1)

    # Convert distances to a probability distribution
    top_k_weights = softmax(-top_k_distances, axis=1)  # Apply softmax to convert to probabilities

    # Step 4: Populate the pseudo-labels array with the probabilities
    for j, indices in enumerate(top_k_indices):
        pseudo_labels[i + j, indices] = top_k_weights[j]

# Save the pseudo-labels
np.save('data/pseudo_labels_ldl.npy', pseudo_labels)

print("Pseudo labels generated successfully")