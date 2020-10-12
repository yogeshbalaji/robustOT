# Code for evaluating trained GAN models.

import torch
import utils
import losses
import os.path as osp
from pathlib import Path
import torchvision.utils as vutils
import math
import numpy as np
import copy
import models
from datasets import dataset_factory
import torch.nn as nn
import os
import json


_ATTR_CLS_MODEL_PATH = '/vulcanscratch/yogesh22/projects/robust_optimal_transport/GAN/attribute_classifier/results/CelebA_attributes_64/model.pth'


class AttributeClassifier(nn.Module):
    expansion = 1

    def __init__(self, nclasses):
        super(AttributeClassifier, self).__init__()
        ndf = 64

        self.feat_net = nn.Sequential(
            nn.Conv2d(3, ndf, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(ndf),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(ndf, 2 * ndf, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(2 * ndf),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(2 * ndf, 4 * ndf, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(4 * ndf),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(4 * ndf, 4 * ndf, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(4 * ndf),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(4 * ndf,  2 * ndf, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(2 * ndf),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(8 * ndf, 2 * ndf),
            nn.ReLU(True),
            nn.Linear(2 * ndf, nclasses)
        )

    def forward(self, x):
        features = self.feat_net(x)
        features = features.view(features.shape[0], -1)
        out = self.classifier(features)
        return out


class Evaluator:
    def __init__(self, config, load_path, save_path):

        self.config = config
        self.device = 'cuda:0'
        self.logdir = os.path.join(save_path, 'evaluation')
        Path(self.logdir).mkdir(parents=True, exist_ok=True)

        if 'weight' in load_path:
            self.weighted_update = True
            print('Weighted')
        else:
            self.weighted_update = False
            print('Unweighted')

        print('Creating generator')
        gen_model_fn = models.generator_factory[config.netG]
        self.netG = gen_model_fn(config)
        self.netG = self.netG.to(self.device)
        self.netG = nn.DataParallel(self.netG)

        print('Loading generator weights')
        generator_weight_path = '{}/model_state.pth'.format(load_path)
        all_weights = torch.load(generator_weight_path)
        self.netG.load_state_dict(all_weights['netG'])

        if self.weighted_update:
            print('Creating weight network')
            weight_model_fn = models.weight_factory[config.netD]
            self.netW = weight_model_fn(config).to(self.device)

        print('Creating data loader')
        attribute_list = [
            ('Male', 1, 0.5),
            ('Male', -1, 0.5)
        ]

        dset_fn = dataset_factory[config.dataset]
        self.dataloader, self.num_classes = dset_fn(
            config.dataroot, 128,
            imgSize=config.imageSize,
            input_attribute_list=attribute_list,
            anomaly_frac=None,
            anomalypath=None,
            savepath=None,
            train=True
        )

        print('Creating attribute classifier')

        self.attribute_classifier = AttributeClassifier(nclasses=2)
        all_state = torch.load(_ATTR_CLS_MODEL_PATH)

        self.attribute_classifier.load_state_dict(all_state['net'])
        self.attribute_classifier = self.attribute_classifier.to(self.device)

    def create_samples(self, num_samples):

        print('Creating samples')
        bs = 100

        gen_list = []
        with torch.no_grad():
            for i in range(int(num_samples / bs)):
                z_cur = utils.sample_normal(bs, self.config.nz, device=self.device)
                y_cur = None

                gen = self.netG(z_cur, y_cur)
                gen_list.append(gen.detach().cpu())

        gen = torch.cat(gen_list, dim=0)
        vutils.save_image(gen[0:100] * 0.5 + 0.5,
                          '{}/samples.png'.format(self.logdir), nrow=10
                          )
        print('Sample creation done')

        return gen

    def evaluate_attributes(self, samples):

        print('Evaluating attributes')
        num_samples = samples.shape[0]
        eval_bs = 100
        num_batches = int(num_samples / eval_bs)

        num_males = 0
        num_females = 0
        with torch.no_grad():
            for i in range(num_batches):
                batch = samples[i * eval_bs: (i+1) * eval_bs, ::]
                batch = batch.to(self.device)

                batch_us = nn.Upsample(scale_factor=2, mode='nearest')(batch)

                logits = self.attribute_classifier(batch_us)
                _, pred = torch.max(logits, dim=1)

                num_males += (pred == 0).sum()
                num_females += (pred == 1).sum()

        male_female_ratio = float(num_males) / (float(num_males) + float(num_females))
        print('Attribute evaluation done')
        return male_female_ratio

    def generate_weights(self):

        print('Generating weights')
        sample_array = []
        weight_array = []

        num_males = 0
        num_females = 0

        with torch.no_grad():
            for i, dat in enumerate(self.dataloader):
                if i > 200:
                    break

                inp, labels, _ = dat
                inp = inp.to(self.device)

                inp_us = nn.Upsample(scale_factor=2, mode='nearest')(inp)

                logits = self.attribute_classifier(inp_us)
                _, pred = torch.max(logits, dim=1)

                num_males += (pred == 0).sum()
                num_females += (pred == 1).sum()

                weight_batch = self.netW(inp)
                weight_batch = weight_batch.detach().cpu()
                sample_array.append(inp.cpu())
                weight_array.append(weight_batch)

        male_female_ratio = float(num_males) / (float(num_males) + float(num_females))
        print('True ratio: {}'.format(male_female_ratio))

        sample_array = torch.cat(sample_array, dim=0)
        weight_array = torch.cat(weight_array, dim=0)
        weight_array = weight_array.view(-1)

        weight_sort_indices = torch.argsort(weight_array)
        num_samples_wt = weight_array.shape[0]
        num_thresh = int(num_samples_wt * 0.05)
        print(num_samples_wt, num_thresh)

        indices_low = weight_sort_indices[0:num_thresh]
        indices_high = weight_sort_indices[-num_thresh:]

        ind_sel = torch.randperm(num_thresh)[0:50]
        indices_low = indices_low[ind_sel]
        ind_sel = torch.randperm(num_thresh)[0:50]
        indices_high = indices_high[ind_sel]

        samples_low = sample_array[indices_low]
        samples_high = sample_array[indices_high]

        vutils.save_image(
            samples_low * 0.5 + 0.5,
            '{}/samples_low_weights.png'.format(self.logdir), nrow=10
        )
        vutils.save_image(
            samples_high * 0.5 + 0.5,
            '{}/samples_high_weights.png'.format(self.logdir), nrow=10
        )
        print('Weight generation done')

    def eval(self):
        samples_gen = self.create_samples(num_samples=10000)
        ratio = self.evaluate_attributes(samples_gen)
        with open('{}/eval_log_stats.txt'.format(self.logdir), 'w') as fp:
            line = 'Estimated ratio: {}\n'.format(ratio)
            fp.write(line)
            print(line)

        if self.weighted_update:
            self.generate_weights()


def main():

    eval_root = 'results/unconditional/WGAN/CelebA_attributes'
    save_root = 'results/evaluation_WGAN'
    Path(save_root).mkdir(exist_ok=True, parents=True)

    folders = os.listdir(eval_root)
    for fol in folders:
        print('Evaluating {}'.format(fol))
        load_path = os.path.join(eval_root, fol)
        save_path = os.path.join(save_root, fol)
        config = json.load(open('{}/config.json'.format(load_path), 'r'))

        # General args
        config = utils.ConfigMapper(config)
        config.imageSize = 32
        config.num_classes = 2
        config.dataset = 'celeba_attribute'
        config.dataroot = '/vulcanscratch/yogesh22/data/celebA/'
        config.G_bias = True

        evaluator_class = Evaluator(config, load_path, save_path)
        evaluator_class.eval()
        print('#########')


if __name__ == '__main__':
    main()