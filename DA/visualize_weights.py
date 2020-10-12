# Code for visualizing weights

from __future__ import print_function

import argparse
import datasets
import torch
import os
import json
from PIL import Image
import numpy as np
import torchvision.utils as vutils
import utils
from pathlib import Path
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Training settings
def parse_args():
    parser = argparse.ArgumentParser(description='Visualization script')
    parser.add_argument('--cfg-path', type=str, required=True,
                        help='config path')
    parser.add_argument('--results-path', type=str, required=True,
                        help='Results path')
    args = parser.parse_args()
    return args


def read_images(paths):
    transformer = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    img_all = []
    for pth in paths:
        img = Image.open(pth)
        img = transformer(img)
        img_all.append(img)
    
    img_all = torch.stack(img_all)
    return img_all


def visualize(args):

    num_vis = 100

    # Forming config
    with open(args.cfg_path) as json_file:
        config = json.load(json_file)
    config = utils.ConfigMapper(config)

    # Create dataloader
    source_loader, target_loader, nclasses = datasets.form_visda_datasets(config=config, ignore_anomaly=True)

    # Loading model state
    model_state = torch.load(os.path.join(args.results_path, 'model_state.pth'))

    weight_vector = model_state['weight_vector']
    indices_sorted = torch.argsort(weight_vector)
    num = weight_vector.shape[0]
    sampling_interval = int(num / num_vis)
    indices_sampled = indices_sorted[0:num:sampling_interval]

    path_vector_all = target_loader.dataset.samples
    paths = []
    for ind in indices_sampled:
        paths.append(path_vector_all[ind][0])
        print(weight_vector[ind])

    imgs = read_images(paths)
    vutils.save_image(imgs, '{}/weight_vis.png'.format(args.results_path), nrow=10)

    weight_vector_np = weight_vector.cpu().numpy()
    plt.figure()
    plt.rcParams.update({'font.size': 19})
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.hist(weight_vector_np, bins=200)
    plt.xlabel('Weight')
    plt.ylabel('Count')
    plt.yticks([0, 1000, 2000, 3000, 4000, 5000, 6000])
    plt.savefig('{}/weight_hist.png'.format(args.results_path), dpi=300)

    


if __name__ == '__main__':
    args = parse_args()
    visualize(args)
