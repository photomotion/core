## import package
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms

import os
import numpy as np
import itertools
import matplotlib.pyplot as plt
import argparse

from data_processing.DataLoader import *
from Model.model import *
from utils import *

def parse_arguments():
    parser = argparse.ArgumentParser(description="CycleGAN", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--data_dir", action="store", default="./datasets/monet2photo", type=str, dest="data_dir", help="path for the images directory")
    parser.add_argument("--ckpt_dir", action="store", default="./checkpoint", type=str, dest="ckpt_dir", help="saved checkpoint for model weights ")
    parser.add_argument("--result_dir", action="store", default="./result/train", type=str, dest="result_dir", help="save image result")
    parser.add_argument("--img_size", action="store", default=256, type=int, dest="img_size", help="image size")
    parser.add_argument("--img_channels", action="store", default=3, type=int, dest="img_channels", help="Channel in input images")
    parser.add_argument("--nker", action="store", default=64, type=int, dest="nker", help="number of kernel size")
    parser.add_argument("--norm", action="store", default="inorm", type=str, dest="norm", help="Normalization [bnorm:batch norm, inorm: instanceNorm]")
    parser.add_argument("--lr", action="store", default=2e-4, type=float, dest="lr", help="learning rate")
    parser.add_argument("--batch_size", action="store", default=4, type=int, dest="batch_size", help="batch size")
    parser.add_argument("--start_epoch", action="store", default=1, type=int, dest="start_epoch", help="starting epoch number")
    parser.add_argument("--num_epoch", action="store", default=100, type=int, dest="num_epoch", help="number of epochs for training")
    parser.add_argument("--wgt_cycle", action="store", default=10, type=int, dest="wgt_cycle", help="weight for cycle consistency loss")
    parser.add_argument("--wgt_ident", action="store", default=0.5, type=int, dest="wgt_ident", help="identity mapping loss of weight")
    parser.add_argument("--num_workers", action="store", default=16, type=int, dest="num_workers", help="number of parallel workers for reading files")
    parser.add_argument("--device", action="store", default="torch.device('cuda' if torch.cuda.is_available() else 'cpu')", type=str, dest="device", help="whether GPU or CPU")

    args = parser.parse_args()
    print('args={}'.format(args))

    return args

def main(args):
    ## 디렉토리 생성하기
    if not os.path.exists(args.result_dir):
        os.makedirs(os.path.join(args.result_dir, 'png', 'a2b'))
        os.makedirs(os.path.join(args.result_dir, 'png', 'b2a'))

    ## Data Loader 설정
    transform = transforms.Compose([
        Resize(shape=(286, 286, args.img_channels)),
        RandomCrop(shape=(args.img_size, args.img_size)),
        Normalization(mean=0.5, std=0.5)
    ])

    dataset = Dataset(data_dir=os.path.join(args.data_dir, 'train'), transform=transform, data_type='both')
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    ## 네트워크 설정
    netG_a2b = CycleGAN(in_channels=args.img_channels, out_channels=args.img_channels, nker=args.nker, norm=args.norm, nblock=9).to(args.device)
    netG_b2a = CycleGAN(in_channels=args.img_channels, out_channels=args.img_channels, nker=args.nker, norm=args.norm, nblock=9).to(args.device)
    netD_a = Discriminator(in_channels=args.img_channels, out_channels=1, nker=args.nker, norm=args.norm).to(args.device)
    netD_b = Discriminator(in_channels=args.img_channels, out_channels=1, nker=args.nker, norm=args.norm).to(args.device)

    weights_init_normal(net=netG_a2b, init_type='normal', init_gain=0.02)
    weights_init_normal(net=netG_b2a, init_type='normal', init_gain=0.02)
    weights_init_normal(net=netD_a,init_type='normal', init_gain=0.02)
    weights_init_normal(net=netD_b, init_type='normal', init_gain=0.02)

    ## Loss Function 정의
    cycle_loss_fn =nn.L1Loss().to(args.device) # Cycle Loss
    gan_loss_fn = nn.BCELoss().to(args.device) # Adversarial Loss
    ident_loss_fn = nn.L1Loss().to(args.device) # identity function

    ## OPTIMIZER 설정
    optimG = Adam(itertools.chain(netG_a2b.parameters(), netG_b2a.parameters()), lr=args.lr, betas=(0.5, 0.999))
    optimD = Adam(itertools.chain(netD_a.parameters(), netD_b.parameters()), lr=args.lr, betas=(0.5, 0.999))

    for epoch in range(args.start_epoch, args.num_epoch):
        netG_a2b.train(); netG_b2a.train()
        netD_a.train(); netD_b.train()

        for batch, data in enumerate(data_loader, 1):
            input_a = data['img_A'].to(args.device)
            input_b = data['img_B'].to(args.device)

            # ============================================ #
            #            Discriminator Training            #
            # ============================================ #

            # Forward netG
            output_b = netG_a2b(input_a)
            output_a = netG_b2a(input_b)

            recon_b = netG_a2b(output_a)
            recon_a = netG_b2a(output_b)

            ## Backward netD
            set_requires_grad(nets=[netD_a, netD_b], requires_grad=True)
            optimD.zero_grad()

            # backward netD_a
            pred_real_a = netD_a(input_a)
            pred_fake_a = netD_a(output_a.detach())

            loss_D_a_real = gan_loss_fn(pred_real_a, torch.ones_like(pred_real_a))
            loss_D_a_fake = gan_loss_fn(pred_fake_a, torch.zeros_like(pred_fake_a))
            loss_D_a = (loss_D_a_real + loss_D_a_fake) / 2

            # backward netD_b
            pred_real_b = netD_b(input_b)
            pred_fake_b = netD_b(output_b.detach())

            loss_D_b_real = gan_loss_fn(pred_real_b, torch.ones_like((pred_real_b)))
            loss_D_b_fake = gan_loss_fn(pred_fake_b, torch.zeros_like((pred_fake_b)))
            loss_D_b = (loss_D_b_real + loss_D_b_fake) / 2

            # Total loss D
            loss_D = loss_D_a + loss_D_b
            loss_D.backward()
            optimD.step()


            # ============================================ #
            #            Generator Training                #
            # ============================================ #
            # Backward netG
            set_requires_grad(nets=[netD_a, netD_b], requires_grad=False)
            optimG.zero_grad()

            pred_fake_a = netD_a(output_a)
            pred_fake_b = netD_b(output_b)

            # Adversarial loss
            loss_G_a2b = gan_loss_fn(pred_fake_a, torch.ones_like(pred_fake_a))
            loss_G_b2a = gan_loss_fn(pred_fake_b, torch.ones_like(pred_fake_b))

            # Cycle loss
            loss_cycle_a = cycle_loss_fn(input_a, recon_a)
            loss_cycle_b = cycle_loss_fn(input_b, recon_b)

            # identity loss
            loss_ident_a = ident_loss_fn(input_a, netG_b2a(input_a))
            loss_ident_b = ident_loss_fn(input_b, netG_a2b(input_b))

            loss_G = (loss_G_a2b + loss_G_b2a) + \
                     args.wgt_cycle * (loss_cycle_a + loss_cycle_b) + \
                     args.wgt_cycle * args.wgt_ident * (loss_ident_a + loss_ident_b)

            loss_G.backward()
            optimG.step()






















if __name__ == '__main__':
    # invoke the main function of the script
    main(parse_arguments())


