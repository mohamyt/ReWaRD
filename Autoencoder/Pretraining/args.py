import argparse
import sys

def conf(args_list=None):
    parser = argparse.ArgumentParser(description="PyTorch Jigsaw Pretext")
    # model name
    parser.add_argument("--dataset", default="ImageNet1k", type = str, help="dataset name")
    # network settings
    parser.add_argument("--usenet", default="resnet18", type = str, help="use network")
    parser.add_argument("--epochs", default=90, type = int, help="end epoch")
    parser.add_argument("--numof_classes", default=1000, type = int, help="num of classes")
    parser.add_argument("--model_type", default="Autoencoder", type = str, help="use network")
    # model hyper-parameters
    parser.add_argument("--lr", default=0.0001, type = float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type = float, help="momentum")
    parser.add_argument("--weight_decay", default=1e-4, type = float, help="weight decay")
    # scheduler parameters
    parser.add_argument("--scheduler_milestones", default=[90, 150, 200, 250], type=int, nargs='+', help="epochs at which to decay learning rate")
    parser.add_argument("--scheduler_gamma", default=0.2, type=float, help="learning rate decay factor")
    parser.add_argument('--resume_scheduler', default=True, action='store_true', help='If true, resume scheduler from checkpoint.')
    # etc
    parser.add_argument("--start-epoch", default=1, type = int, help="input batch size for training")
    parser.add_argument("--batch_size", default=64, type = int, help="input batch size for training")
    parser.add_argument("--val-batch_size", default=64, type=int, help="input batch size for testing")
    parser.add_argument("--img_size", default=256, type = int, help="image size")
    parser.add_argument("--crop_size", default=256, type = int, help="crop size")
    parser.add_argument('--no_multigpu', default=False, action='store_true', help='If true, training is not performed.')
    parser.add_argument("--no-cuda", default=False, action="store_true", help="disables CUDA training")
    parser.add_argument("--gpu_id", default=-1, type = int, help="gpu id")
    parser.add_argument("--num_workers", default=8, type = int, help="num of workers (data_loader)")
    parser.add_argument("--save-interval", default=1, type = int, help="save every N epoch")
    parser.add_argument("--seed", default=1, type=int, help="seed")
    parser.add_argument('--lmdb', default=False, action='store_true', help='If true, training database is an lmdb file.')
    
    if args_list is not None:
        args, unknown = parser.parse_known_args(args_list)
    else:
        args, unknown = parser.parse_known_args()

    # paths
    parser.add_argument('--val', default=True, action='store_true', help='If true, training is not performed.')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument("--path2traindb", default="./data/ImageNet1k/train", type = str, help="path to dataset training images") #dataset path 
    parser.add_argument("--path2valdb", default="./data/ImageNet1k/val", type = str, help="path to dataset validation images")

    if args_list is not None:
        args, unknown = parser.parse_known_args(args_list)
    else:
        args, unknown = parser.parse_known_args()
    
    return args

if __name__ == "__main__":
    args = conf(sys.argv[1:])
    print(args)