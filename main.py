import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import warnings
warnings.filterwarnings("ignore")

from src.neil.dataset import prepare_dataset_for_training
from src.neil.train import train


def get_parser():
    parser = ArgumentParser('Dual attention attack')
    commands = parser.add_subparsers(dest='cmd')
    
    prepare_dataset = commands.add_parser('prepare', formatter_class=ArgumentDefaultsHelpFormatter)
    prepare_dataset.add_argument('-b', '--batch-size', type=int, default=1, help='The batch size')
    prepare_dataset.add_argument('-dist', '--distance', type=int, default=-1, help='The distance to move the camera by')
    prepare_dataset.add_argument('-t', '--texture-size', type=int, default=6, help='Size of the texture')
    prepare_dataset.add_argument('-i', '--image-size', type=int, default=800, help='Image size')
    prepare_dataset.add_argument('-d', '--dataset', type=str, default='./dataset', help='Location of the folder containing "phy_attack"')
    prepare_dataset.add_argument('-o', '--output', type=str, default='./dataset', help='Location where the prepared images are to be saved')
    prepare_dataset.add_argument('-obj', '--vehicle-object', type=str,
                                 default='./assets/object_files/audi/audi_et_te.obj', help='The OBJ file to apply on the dataset')
    
    train = commands.add_parser('train', formatter_class=ArgumentDefaultsHelpFormatter)
    train.add_argument('-obj', '--vehicle-object', type=str,
                                 default='./assets/object_files/audi/audi_et_te.obj', help='The OBJ file to apply on the dataset')
    train.add_argument('-t', '--texture-size', type=int, default=6, help='Size of the texture')
    train.add_argument('-e', '--epochs', type=int, default=1, help='The number of epochs to train the model')
    train.add_argument('-b', '--batch-size', type=int, default=1, help='The batch size')
    train.add_argument('-mm', '--masks_dir', type=str, default='./src/data', help='Location of the folder containing "masks"')
    train.add_argument('-d', '--dataset', type=str, default='/home/nsambhu/DAS_output', help='Location of the folder containing "rendering" and "mash"')
    train.add_argument('-c', '--content-src', type=str, default='./assets/contents/smile.jpg', help='Content source"')
    train.add_argument('-cn', '--canny-src', type=str, default='assets/contents/smile_edge.jpg', help='Canny source"')
    train.add_argument('-i', '--image-size', type=int, default=800, help='Image size')
    train.add_argument('-ce', '--cam-edge', type=int, default=7, help='Camera edge')
    train.add_argument('-d1', '--d1', type=float, default=0.9, help='d1')
    train.add_argument('-d2', '--d2', type=float, default=0.1, help='d2')
    train.add_argument('-tt', '--t', type=float, default=0.0001, help='d2')
    train.add_argument('-s', '--save-every', type=int, default=1000, help='Save the model every so many batches')
    train.add_argument('-m', '--model-dst', type=str, default='./models', help='Location to save the model')
       
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    
    if args.cmd == 'prepare':
        print('Preparing the dataset...')
        prepare_dataset_for_training(args.dataset, args.output, args.vehicle_object, 
                                     args.batch_size, args.image_size, 
                                     args.texture_size, args.distance)
    
    elif args.cmd == 'train':
        print('Training the model...')
        os.makedirs(args.model_dst, exist_ok=True)
        train(args)

