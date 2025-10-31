import argparse
import sys

def conf(args_list=None):
    parser = argparse.ArgumentParser(description="PyTorch Jigsaw Pretext")
    # model name
    parser.add_argument("--dataset", default="ImageNet1k", type = str, help="dataset name")
    # network settings
    parser.add_argument("--usenet", default="resnet18", type = str, help="use network")
    parser.add_argument("--model_type", default="LDL", type = str, help="Pretrained model type")
    parser.add_argument("--epochs", default=250, type = int, help="end epoch")
    parser.add_argument("--numof_classes", default=1000, type = int, help="num of classes")
    # model hyper-parameters
    parser.add_argument("--lr", default=0.1, type = float, help="initial learning rate")
    parser.add_argument('--use_last_lr', default=True, action='store_true', help='If true, training learning rate starts from the last learning rate used in checkpoint and lr variable is ignored.')
    parser.add_argument("--momentum", default=0.9, type = float, help="momentum")
    parser.add_argument("--weight_decay", default=1e-4, type = float, help="weight decay")
    
    # scheduler parameters
    parser.add_argument('--resume_scheduler', default=True, action='store_true', help='If true, scheduler is resumed relative to the number of epochs performed on the resumed model.')
    parser.add_argument("--scheduler_milestones", default=[30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 390, 420, 450, 480, 510, 540, 570, 600, 630, 660, 690, 720, 750, 780, 810, 840, 870, 900, 930, 960, 990], type=int, nargs='+', help="epochs at which to decay learning rate")
    parser.add_argument("--scheduler_gamma", default=0.1, type=float, help="learning rate decay factor")
    # transform parameters
    parser.add_argument("--r_crop_size", default=224, type = int, help="random crop size")
    parser.add_argument("--p_grayscale", default=0.2, type = float, help="random greyscale probability")
    parser.add_argument("--p_blur", default=0.2, type = float, help="random blur probability")
    parser.add_argument("--min_blur_r", default=2, type = int, help="minimum radius of random blur")
    parser.add_argument("--max_blur_r", default=4, type = int, help="maximum radius of random blur")
    # etc
    parser.add_argument("--start-epoch", default=1, type = int, help="input batch size for training")
    parser.add_argument("--batch_size", default=256, type = int, help="input batch size for training")
    parser.add_argument("--val-batch_size", default=256, type=int, help="input batch size for testing")
    parser.add_argument("--img_size", default=256, type = int, help="image size")
    parser.add_argument("--crop_size", default=256, type = int, help="crop size")
    parser.add_argument('--no_multigpu', default=False, action='store_true', help='If true, training is not performed.')
    parser.add_argument("--no-cuda", default=False, action="store_true", help="disables CUDA training")
    parser.add_argument("--gpu_id", default=-1, type = int, help="gpu id")
    parser.add_argument("--num_workers", default=8, type = int, help="num of workers (data_loader)")
    parser.add_argument("--save-interval", default=10, type = int, help="save every N epoch")
    parser.add_argument("--seed", default=1, type=int, help="seed")
    parser.add_argument('--lmdb', default=False, action='store_true', help='If true, training database is an lmdb file.')
    parser.add_argument('--freeze_pretrained', action='store_true', default=True, help='If true, the backbone is frozen and only the classifier is trained.')
    parser.add_argument('--freeze_epoch', type=int, default=5, help='Freeze backbone after this many epochs (1-based). Use 0 to freeze immediately. If None, backbone is never frozen.')

    
    if args_list is not None:
        args, unknown = parser.parse_known_args(args_list)
    else:
        args, unknown = parser.parse_known_args()

    # paths
    parser.add_argument('--val', default=True, action='store_true', help='If true, training is not performed.')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument("--path2model", default=f"/home/hpc/iwi9/iwi9120h/ReWaRD-Unsupervised-Learning/Finetuning/data/weight/Pretrained/LDL_resnet18.tar", type = str, help="path to dataset training images")
    parser.add_argument("--path2traindb", default="/home/hpc/iwi9/iwi9120h/.cache/kagglehub/datasets/amitpant7/imagenet1k-subset-100k-train-and-10k-val/versions/2/imagenet_subtrain", type = str, help="path to dataset training images")
    parser.add_argument("--path2valdb", default="/home/hpc/iwi9/iwi9120h/.cache/kagglehub/datasets/amitpant7/imagenet1k-subset-100k-train-and-10k-val/versions/2/imagenet_subval", type = str, help="path to dataset validation images")


    if args_list is not None:
        args, unknown = parser.parse_known_args(args_list)
    else:
        args, unknown = parser.parse_known_args()
    
    return args

if __name__ == "__main__":
    args = conf(sys.argv[1:])
    print(args)
