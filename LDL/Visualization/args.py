import argparse
import sys

def conf(args_list=None):
    
    parser = argparse.ArgumentParser(description="PyTorch Jigsaw Pretext")
    # model name
    parser.add_argument("--dataset", default="rwave-1024", type = str, help="dataset name")
    # network settings
    parser.add_argument("--usenet", default="resnet18", type = str, help="use network")
    parser.add_argument("--epochs", default=100, type = int, help="end epoch")
    parser.add_argument("--numof_classes", default=128, type = int, help="num of classes")
    # model hyper-parameters
    parser.add_argument("--lr", default=0.07, type = float, help="initial learning rate")
    parser.add_argument('--use_last_lr', default=True, action='store_true', help='If true, training learning rate starts from the last learning rate used in checkpoint and lr variable is ignored.')
    parser.add_argument("--momentum", default=0.9, type = float, help="momentum")
    parser.add_argument("--weight_decay", default=1e-4, type = float, help="weight decay")
    parser.add_argument("--out_dim", default=128, type = int, help="output feature vector size after the projection head")
    parser.add_argument("--temperature", default=0.5, type = float, help="NT-Xent loss function temperature parameter")
    parser.add_argument("--K", default=5, type = int, help="K nearest neighbors for KNN used in assigning pseudo labels")
    
    # scheduler parameters
    parser.add_argument("--scheduler_milestones", default=[10, 15, 20, 30, 35, 40, 45, 50, 55, 60], type=int, nargs='+', help="epochs at which to decay learning rate")
    parser.add_argument("--scheduler_gamma", default=0.1, type=float, help="learning rate decay factor")
    # transform parameters
    parser.add_argument("--r_crop_size", default=224, type = int, help="random crop size")
    parser.add_argument("--p_grayscale", default=0.2, type = float, help="random greyscale probability")
    parser.add_argument("--p_blur", default=0.2, type = float, help="random blur probability")
    parser.add_argument("--min_blur_r", default=2, type = int, help="minimum radius of random blur")
    parser.add_argument("--max_blur_r", default=4, type = int, help="maximum radius of random blur")
    # etc
    parser.add_argument("--start-epoch", default=1, type = int, help="input batch size for training")
    parser.add_argument("--batch_size", default=512, type = int, help="input batch size for training")
    parser.add_argument("--val-batch_size", default=512, type=int, help="input batch size for testing")
    parser.add_argument("--img_size", default=256, type = int, help="image size")
    parser.add_argument("--crop_size", default=256, type = int, help="crop size")
    parser.add_argument('--no_multigpu', default=False, action='store_true', help='If true, training is not performed.')
    parser.add_argument("--no-cuda", default=False, action="store_true", help="disables CUDA training")
    parser.add_argument("--gpu_id", default=-1, type = int, help="gpu id")
    parser.add_argument("--num_workers", default=8, type = int, help="num of workers (data_loader)")
    parser.add_argument("--save-interval", default=5, type = int, help="save every N epoch")
    parser.add_argument("--seed", default=1, type=int, help="seed")
    parser.add_argument('--lmdb', default=True, action='store_true', help='If true, training database is an lmdb file.')
    
    if args_list is not None:
        args, unknown = parser.parse_known_args(args_list)
    else:
        args, unknown = parser.parse_known_args()

    # paths
    parser.add_argument('--val', default=True, action='store_true', help='If true, training is not performed.')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument("--path2traindb", default="./data/rwave-1024/rwave-1024_labelled.lmdb", type = str, help="path to dataset training images") #dataset path 
    parser.add_argument("--path2valdb", default="./data/rwave-1024/rwave-1024_labelled.lmdb", type = str, help="path to dataset validation images")


    if args_list is not None:
        args, unknown = parser.parse_known_args(args_list)
    else:
        args, unknown = parser.parse_known_args()
    
    return args

if __name__ == "__main__":
    args = conf(sys.argv[1:])
    print(args)