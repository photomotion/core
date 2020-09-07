## import package
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

import os
import numpy as np
import itertools
import matplotlib.pyplot as plt
import argparse

from data_processing.DataLoader import *

def parse_arguments():
    parser = argparse.ArgumentParser(description="CycleGAN", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--data_dir", action="store", default="./datasets/monet2photo", type=str, dest="data_dir", help="path for the images directory")
    parser.add_argument("--ckpt_dir", action="store", default="./checkpoint", type=str, dest="ckpt_dir", help="saved checkpoint for model weights ")
    parser.add_argument("--result_dir", action="store", default="./result/train", type=str, dest="result_dir", help="save image result")
    parser.add_argument("--img_size", action="store", default="256", type=int, dest="img_size", help="image size")
    parser.add_argument("--img_channels", action="store", default="3", type=int, dest="img_channels", help="Channel in input images")
    parser.add_argument("--lr", action="store", default="2e-4", type=float, dest="lr", help="learning rate")
    parser.add_argument("--batch_size", action="store", default="4", type=int, dest="batch_size", help="batch size")
    parser.add_argument("--num_epoch", action="store", default="100", type=int, dest="num_epoch", help="number of epochs for training")
    parser.add_argument("--num_workers", action="store", default="16", type=int, dest="num_workers", help="number of parallel workers for reading files")
    parser.add_argument("--device", action="store", default="torch.device('cuda' if torch.cuda.is_available() else 'cpu')", type=str, dest="device", help="whether GPU or CPU")

    args = parser.parse_args()
    print('args={}'.format(args))

    return args



def main(args):
    ## 디렉토리 생성하기
    if not os.path.exists(args.result_dir):
        os.makedirs(os.path.join(args.result_dir, 'png', 'a2b'))
        os.makedirs(os.path.join(args.result_dir, 'png', 'b2a'))

    ## 네트워크 학습하기
    transform = transforms.Compose([
        Resize(shape=(286, 286, args.img_channels)),
        RandomCrop(shape=(args.img_size, args.img_size)),
        Normalization(mean=0.5, std=0.5)
    ])

    dataset = Dataset(data_dir=os.path.join(args.data_dir, 'train'), transform=transform, data_type='both')
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)






if __name__ == '__main__':
    # invoke the main function of the script
    main(parse_arguments())


