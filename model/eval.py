# Copyright 2019 The IEVA-DGM Authors. All rights reserved.
# Use of this source code is governed by a MIT-style license that can be
# found in the LICENSE file.

# model evaluation

from __future__ import absolute_import, division, print_function

import os
import argparse
import math

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage import color
# from skimage.measure import compare_ssim
# from sklearn.metrics.pairwise import euclidean_distances
# import pyemd
import imageio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

import sys
sys.path.append("..")
sys.path.append("../datasets")
sys.path.append("../model")

from mpas import *
from generator import Generator
from discriminator import Discriminator
from vgg19 import VGG19
# from plot_style import default

# example usage:
# python3 eval.py --root /Users/rhythm/Desktop/nyx/ --resume ../models/nyx/perc_gan48/model_False_relu1_2_vanilla_100.pth.tar --ch 48 --sn --perc-loss relu1_2 --gan-loss vanilla --gan-loss-weight 0.01 --batch-size 1

# parse arguments
def parse_args():
  parser = argparse.ArgumentParser(description="Deep Learning Model")

  parser.add_argument("--no-cuda", action="store_true", default=False,
                      help="disables CUDA training")
  parser.add_argument("--data-parallel", action="store_true", default=False,
                      help="enable data parallelism")
  parser.add_argument("--seed", type=int, default=1,
                      help="random seed (default: 1)")

  parser.add_argument("--root", required=True, type=str,
                      help="root of the dataset")
  parser.add_argument("--resume", type=str, default="",
                      help="path to the latest checkpoint (default: none)")

  parser.add_argument("--dsp", type=int, default=3,
                      help="dimensions of the simulation parameters (default: 3)")
  parser.add_argument("--dvo", type=int, default=3,
                      help="dimensions of the visualization operations (default: 3)")                   
  parser.add_argument("--dvp", type=int, default=3,
                      help="dimensions of the visualization parameters (default: 3)")
  parser.add_argument("--dspe", type=int, default=512,
                      help="dimensions of the simulation parameters' encode (default: 512)")
  parser.add_argument("--dvoe", type=int, default=512,
                      help="dimensions of the visualization operations' encode (default: 512)")
  parser.add_argument("--dvpe", type=int, default=512,
                      help="dimensions of the visualization parameters' encode (default: 512)")
                    
  parser.add_argument("--ch", type=int, default=64,
                      help="channel multiplier (default: 64)")

  parser.add_argument("--sn", action="store_true", default=False,
                      help="enable spectral normalization")

  parser.add_argument("--mse-loss", action="store_true", default=False,
                      help="enable mse loss")
  parser.add_argument("--perc-loss", type=str, default="relu1_2",
                      help="layer that perceptual loss is computed on (default: relu1_2)")
  parser.add_argument("--gan-loss", type=str, default="none",
                      help="gan loss (default: none)")
  parser.add_argument("--gan-loss-weight", type=float, default=0.,
                      help="weight of the gan loss (default: 0.)")

  parser.add_argument("--lr", type=float, default=1e-3,
                      help="learning rate (default: 1e-3)")
  parser.add_argument("--d-lr", type=float, default=1e-3,
                      help="learning rate of the discriminator (default: 1e-3)")
  parser.add_argument("--beta1", type=float, default=0.9,
                      help="beta1 of Adam (default: 0.9)")
  parser.add_argument("--beta2", type=float, default=0.999,
                      help="beta2 of Adam (default: 0.999)")
  parser.add_argument("--batch-size", type=int, default=50,
                      help="batch size for training (default: 50)")
  parser.add_argument("--start-epoch", type=int, default=0,
                      help="start epoch number (default: 0)")
  parser.add_argument("--epochs", type=int, default=10,
                      help="number of epochs to train (default: 10)")

  parser.add_argument("--log-every", type=int, default=10,
                      help="log training status every given number of batches (default: 10)")
  parser.add_argument("--check-every", type=int, default=20,
                      help="save checkpoint every given number of epochs (default: 20)")

  parser.add_argument("--id", type=int, default=0,
                      help="index of the data to evaluate (default: 0)")

  return parser.parse_args()

# the main function
def main(args):
  # log hyperparameters
  print(args)

  # select device
  args.cuda = not args.no_cuda and torch.cuda.is_available()
  device = torch.device("cuda:0" if args.cuda else "cpu")

  # set random seed
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)

  # data loader
  train_dataset = MPASDataset(
      root=args.root,
      train=True,
      transform=transforms.Compose([Normalize(), ToTensor()]))

  test_dataset = MPASDataset(
      root=args.root,
      train=False,
      transform=transforms.Compose([Normalize(), ToTensor()]))

  kwargs = {"num_workers": 4, "pin_memory": True} if args.cuda else {}
  train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                            shuffle=False, **kwargs)
  test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                           shuffle=False, **kwargs)

  # model
  def weights_init(m):
    if isinstance(m, nn.Linear):
      nn.init.orthogonal_(m.weight)
      if m.bias is not None:
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
      nn.init.orthogonal_(m.weight)
      if m.bias is not None:
        nn.init.zeros_(m.bias)

  def add_sn(m):
    for name, c in m.named_children():
      m.add_module(name, add_sn(c))
    if isinstance(m, (nn.Linear, nn.Conv2d)):
      return nn.utils.spectral_norm(m, eps=1e-4)
    else:
      return m

  g_model = Generator(dsp=args.dsp, dvo=args.dvo, dvp=args.dvp,
                      dspe=args.dspe, dvoe=args.dvoe, dvpe=args.dvpe,
                      ch=args.ch)
  g_model.apply(weights_init)
  # if args.sn:
  #   g_model = add_sn(g_model)

  if args.data_parallel and torch.cuda.device_count() > 1:
    g_model = nn.DataParallel(g_model)
  g_model.to(device)

  # if args.gan_loss != "none":
  #   d_model = Discriminator(args.dsp, args.dvp, args.dspe,
  #                           args.dvpe, args.ch)
  #   d_model.apply(weights_init)
  #   if args.sn:
  #     d_model = add_sn(d_model)

  #   if args.data_parallel and torch.cuda.device_count() > 1:
  #     d_model = nn.DataParallel(d_model)
  #   d_model.to(device)

  # loss
  if args.perc_loss != "none":
    norm_mean = torch.tensor([.485, .456, .406]).view(-1, 1, 1).to(device)
    norm_std = torch.tensor([.229, .224, .225]).view(-1, 1, 1).to(device)
    vgg = VGG19(args.perc_loss).eval()
    if args.data_parallel and torch.cuda.device_count() > 1:
      vgg = nn.DataParallel(vgg)
    vgg.to(device)

  mse_criterion = nn.MSELoss()
  train_losses, test_losses = [], []
  d_losses, g_losses = [], []

  # # optimizer
  # g_optimizer = optim.Adam(g_model.parameters(), lr=args.lr,
  #                          betas=(args.beta1, args.beta2))
  # if args.gan_loss != "none":
  #   d_optimizer = optim.Adam(d_model.parameters(), lr=args.d_lr,
  #                            betas=(args.beta1, args.beta2))

  # load checkpoint
  if args.resume:
    if os.path.isfile(args.resume):
      print("=> loading checkpoint {}".format(args.resume))
      checkpoint = torch.load(args.resume, map_location=device)
      args.start_epoch = checkpoint["epoch"]
      g_model.load_state_dict(checkpoint["g_model_state_dict"])
      # g_optimizer.load_state_dict(checkpoint["g_optimizer_state_dict"])
      if args.gan_loss != "none":
        # d_model.load_state_dict(checkpoint["d_model_state_dict"])
        # d_optimizer.load_state_dict(checkpoint["d_optimizer_state_dict"])
        d_losses = checkpoint["d_losses"]
        g_losses = checkpoint["g_losses"]
      train_losses = checkpoint["train_losses"]
      test_losses = checkpoint["test_losses"]
      print("=> loaded checkpoint {} (epoch {})"
          .format(args.resume, checkpoint["epoch"]))

   # 2) plot sample image
   # selected sample images - 561

  g_model.eval()
  with torch.no_grad():
     sample = test_dataset[args.id]
     image = sample["image"].to(device).view(1, 3, 256, 256)
     sparams = sample["sparams"].to(device).view(1, args.dsp)
     vops = sample["vops"].to(device).view(1, args.dvo)
     vparams = sample["vparams"].to(device).view(1, args.dvp)

     fake_image = g_model(sparams, vops, vparams)

     imageio.imwrite("gt.png", ((image.view(3, 256, 256) + 1.) * .5).cpu().numpy().transpose((1, 2, 0)))

     imageio.imwrite("gan.png", ((fake_image.view(3, 256, 256) + 1.) * .5).cpu().numpy().transpose((1, 2, 0)))

     imageio.imwrite("gan_diff.png", torch.mean(torch.pow(image - fake_image, 2).view(3, 256, 256), 0).cpu().numpy())

  # 3) compute PSNR, SSIM, EMD
#   def compute_emd(im1, im2, cost_mat, l_bins=8, a_bins=12, b_bins=12):
#     lab_im1 = color.rgb2lab(im1.astype(np.uint8))
#     lab_im1 = lab_im1.reshape((lab_im1.shape[0] * lab_im1.shape[1], lab_im1.shape[2]))
#     lab_hist_1, _ = np.histogramdd(lab_im1, bins=(l_bins, a_bins, b_bins), range=[[0., 100.], [-86.185, 98.254], [-107.863, 94.482]], normed=False)

#     lab_im2 = color.rgb2lab(im2.astype(np.uint8))
#     lab_im2 = lab_im2.reshape((lab_im2.shape[0] * lab_im2.shape[1], lab_im2.shape[2]))
#     lab_hist_2, _ = np.histogramdd(lab_im2, bins=(l_bins, a_bins, b_bins), range=[[0., 100.], [-86.185, 98.254], [-107.863, 94.482]], normed=False)

#     n_bins = l_bins * a_bins * b_bins
#     lab_hist_1 = lab_hist_1.reshape((n_bins))
#     lab_hist_2 = lab_hist_2.reshape((n_bins))
#     img_res = lab_im1.shape[0]
#     lab_hist_1 /= img_res
#     lab_hist_2 /= img_res
#     return pyemd.emd(lab_hist_1, lab_hist_2, cost_mat)

#   def compute_emd_cost_mat(l_bins=8, a_bins=12, b_bins=12):
#     n_bins = l_bins * a_bins * b_bins
#     index_mat = np.zeros((l_bins, a_bins, b_bins, 3))
#     for idx in range(l_bins):
#       for jdx in range(a_bins):
#         for kdx in range(b_bins):
#           index_mat[idx, jdx, kdx] = np.array([idx, jdx, kdx])
#     index_mat = index_mat.reshape(n_bins, 3)
#     all_dists = euclidean_distances(index_mat, index_mat)
#     return all_dists / np.max(all_dists)

#   emd_cost_mat = compute_emd_cost_mat()

#   sparams = Variable(torch.zeros(1, 3), requires_grad=True).to(device)
#   vparams = Variable(torch.zeros(1, 3), requires_grad=True).to(device)


#   sparams[0, 0] = (float(.136) - .1375) / .0175
#   sparams[0, 1] = (float(.02253) - .0225) / .001
#   sparams[0, 2] = (float(.59) - .7) / .15

#   vparams[0, 0] = np.cos(np.deg2rad(float(211)))
#   vparams[0, 1] = np.sin(np.deg2rad(float(211)))
#   vparams[0, 2] = float(54) / 90.

#   g_model.eval()
#   mse, ssim, emd = 0., 0., 0.
#   with torch.no_grad():
#     for i in tqdm(range(129)):
#       # image = sample["image"].to(device)
#       # sparams = sample["sparams"].to(device)
#       # vparams = sample["vparams"].to(device)

#       sparams[0, 2] = -1. + 1. / 64. * float(i)

#       fake_image = g_model(sparams, vparams)

#       # save fake images for FID computation
#       for j in range(fake_image.shape[0]):
#         save_image(((fake_image[j].cpu() + 1.) * .5),
#                      "tmp/" + str(j + i) + ".png")

#   #     image = image.to("cpu").numpy() \
#   #                  .transpose((0, 2, 3, 1)) * 127.5 + 127.5
#   #     fake_image = fake_image.to("cpu").numpy() \
#   #                            .transpose((0, 2, 3, 1)) * 127.5 + 127.5
#   #     image_size_r = 1. / image.shape[1] / image.shape[2] / image.shape[3]

#   #     # mse
#   #     mse += np.sum(np.power(image - fake_image, 2.)) * image_size_r

#   #     # ssim
#   #     for j in range(image.shape[0]):
#   #       ssim += compare_ssim(image[j], fake_image[j],
#   #                            data_range=255., multichannel=True)

#   #     # color emd
#   #     for j in range(image.shape[0]):
#   #       emd += compute_emd(image[j], fake_image[j], emd_cost_mat)

#   # print("====> PSNR {}, SSIM {}, EMD {}"
#   #     .format(20. * np.log10(255.) -
#   #             10. * np.log10(mse / len(test_loader.dataset)),
#   #             ssim / len(test_loader.dataset),
#   #             emd / len(test_loader.dataset)))

if __name__ == "__main__":
  main(parse_args())