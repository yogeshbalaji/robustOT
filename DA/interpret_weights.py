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
    nclasses = 12
    
    model_state = torch.load(os.path.join(args.results_path, 'model_state.pth'))
    weight_vector = model_state['weight_vector']
    weight_vector = weight_vector.cpu().numpy()

    source_count_list = [0] * nclasses
    target_count_list = [0] * nclasses 
    weight_count_list = [0] * nclasses
    
    source_samples = source_loader.dataset.samples
    for sample in source_samples:
        source_count_list[sample[1]] += 1

    target_samples = target_loader.dataset.samples
    for i, sample in enumerate(target_samples):
        target_count_list[sample[1]] += 1
        weight_count_list[sample[1]] += weight_vector[i]
    
    source_count_list = np.array(source_count_list)
    target_count_list = np.array(target_count_list)
    weight_count_list = np.array(weight_count_list)
    
    source_count_list = source_count_list / np.sum(source_count_list)
    ntarget = np.sum(target_count_list)
    target_count_list = target_count_list / ntarget
    weight_count_list = (weight_count_list / ntarget) * nclasses

    print(source_count_list)
    print(target_count_list)
    print(weight_count_list * target_count_list)


if __name__ == '__main__':
    args = parse_args()
    visualize(args)
