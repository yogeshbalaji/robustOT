# Dataset factory

import torch
import dataset_lib as datasets
import torchvision.transforms as T
import utils
import os
from random import shuffle
from PIL import Image
import numpy as np
import math
import pickle
import random


def resize_and_crop(img, size):
    img_w, img_h = img.size
    scale_factor = max((float(size[0]) / img_w), (float(size[1]) / img_h))
    re_w, re_h = int(math.ceil(img_w * scale_factor)), int(math.ceil(img_h * scale_factor))

    # Resize
    img = img.resize((re_w, re_h))

    # Crop
    if re_h == size[1]:
        start_w = (re_w - re_h) / 2
        end_w = start_w + size[0]
        start_h = 0
        end_h = start_h + size[1]
    else:
        start_w = 0
        end_w = start_w + size[0]
        start_h = (re_h - re_w) / 2
        end_h = start_h + size[1]

    img = img.crop((start_w, start_h, end_w, end_h))
    return img


def cifar(datapath, batchSize, imgSize=32, num_workers=4, pin_memory=True, train=True, anomaly_frac=0,
          anomalypath=None, cifar100=True, savepath='results'):

    if train:
        data_transformer = T.Compose([
            T.Resize(imgSize),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            utils.UniformNoising(min=0, max=1.0 / 128)
        ])
    else:
        data_transformer = T.Compose([
            T.Resize(imgSize),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    if cifar100:
        dset = datasets.CIFAR100(datapath, train=True, download=True,
                                transform=data_transformer)
        num_classes = 100
    else:
        num_classes = 10
        dset = datasets.CIFAR10(datapath, train=True, download=True,
                                transform=data_transformer)

    if anomaly_frac > 0 and anomalypath != 'label_noise':
        # Adding anomalous samples

        assert anomalypath is not None

        if anomalypath == 'uniform_noise':
            num_ano_samples = int(anomaly_frac * dset.data.shape[0])
            ano_images = np.random.uniform(low=0, high=1, size=(num_ano_samples, 32, 32, 3))
            ano_images = ano_images * 255.0
            ano_images = ano_images.astype(np.uint8)
            ano_labels = np.random.randint(0, num_classes, (ano_images.shape[0],))
        else:
            anomaly_sample_list = get_anomaly_list(anomalypath)
            shuffle(anomaly_sample_list)
            num_ano_samples = int(anomaly_frac * dset.data.shape[0])
            ano_samples = anomaly_sample_list[0:num_ano_samples]
            ano_images = []
            for f in ano_samples:
                img = Image.open(f)
                img = resize_and_crop(img, (32, 32))
                img = np.asarray(img)

                if len(img.shape) == 2:
                    img_3 = np.zeros((img.shape[0], img.shape[1], 3), dtype=img.dtype)
                    for i in range(3):
                        img_3[:, :, i] = img
                    img = img_3
                ano_images.append(img)

            ano_images = np.stack(ano_images)
            ano_labels = np.random.randint(0, num_classes, (ano_images.shape[0], ))

        dset.data = dset.data[0:dset.data.shape[0] - num_ano_samples, ::]
        dset.targets = dset.targets[0:len(dset.targets) - num_ano_samples]
        dset.data = np.concatenate((dset.data, ano_images), axis=0)
        dset.targets.extend(ano_labels)

        print('Num samples: {}'.format(dset.data.shape[0]))
        print('Num labels: {}'.format(len(dset.targets)))

    if anomalypath == 'label_noise':
        # Adding label noise to dataset
        indices = np.random.permutation(len(dset.targets))
        num_ano_samples = int(anomaly_frac * dset.data.shape[0])
        indices = indices[0:num_ano_samples]
        for ind in indices:
            dset.targets[ind] = (dset.targets[ind] + random.randint(0, num_classes-1)) % num_classes

    if train:
        save_dict = {}
        save_dict['data'] = dset.data
        save_dict['labels'] = dset.targets
        with open('{}/cifar.pkl'.format(savepath), "wb") as fp:
            pickle.dump(save_dict, fp)

    dataloader = torch.utils.data.DataLoader(dset, batch_size=batchSize, shuffle=True,
                                             num_workers=num_workers, pin_memory=pin_memory)

    return dataloader, num_classes


def cifar10(datapath, batchSize, imgSize=32, num_workers=4, pin_memory=True, train=True, anomaly_frac=0,
            anomalypath=None, savepath='results'):
    return cifar(datapath, batchSize, imgSize=imgSize, num_workers=num_workers, pin_memory=pin_memory,
                 train=train, anomaly_frac=anomaly_frac, anomalypath=anomalypath, cifar100=False,
                 savepath=savepath)


def cifar100(datapath, batchSize, imgSize=32, num_workers=4, pin_memory=True, train=True, anomaly_frac=0,
            anomalypath=None, savepath='results'):
    return cifar(datapath, batchSize, imgSize=imgSize, num_workers=num_workers, pin_memory=pin_memory,
                 train=train, anomaly_frac=anomaly_frac, anomalypath=anomalypath, cifar100=True,
                 savepath=savepath)


def lsun(datapath, batchSize, imgSize=64, num_workers=4, pin_memory=True):
    data_transformer = T.Compose([
        T.Resize(imgSize),
        T.RandomCrop(imgSize),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dset = datasets.LSUN(db_path=datapath, classes=['bedroom_train'],
                         transform=data_transformer)
    dataloader = torch.utils.data.DataLoader(dset, batch_size=batchSize, shuffle=True,
                                             num_workers=num_workers, pin_memory=pin_memory)
    num_classes = 1
    return dataloader, num_classes


def CelebA(datapath, batchSize, imgSize=64, num_workers=4, pin_memory=True,
           anomaly_frac=0, anomalypath=None, savepath='results', train=True):
    data_transformer = T.Compose([
        T.Resize(imgSize),
        T.RandomCrop(imgSize),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dset = datasets.ImageFolder(root=datapath, transform=data_transformer)
    dataloader = torch.utils.data.DataLoader(dset, batch_size=batchSize, shuffle=True,
                                             num_workers=num_workers, pin_memory=pin_memory)
    num_classes = 1
    return dataloader, num_classes


def CelebAWithAttributes(datapath, batchSize, imgSize=64, num_workers=4, pin_memory=True,
                         input_attribute_list=None, anomaly_frac=0, anomalypath=None, savepath='results', train=True):

    # attribute_list is a list of attribute_tuples

    assert input_attribute_list is not None

    data_transformer = T.Compose([
        T.Resize(imgSize),
        T.RandomCrop(imgSize),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dset = datasets.CelebAAttributes(root=datapath, input_attribute_list=input_attribute_list,
                                     transform=data_transformer)
    dataloader = torch.utils.data.DataLoader(dset, batch_size=batchSize, shuffle=True,
                                             num_workers=num_workers, pin_memory=pin_memory)
    num_classes = len(input_attribute_list)
    return dataloader, num_classes


def CelebAHQ(datapath, batchSize, imgSize, num_workers=4, pin_memory=True):
    filename = datapath.split('/')[-1]
    assert str(imgSize) in filename

    data_transformer = T.Compose([
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dset = datasets.ImageFolder(root=datapath, transform=data_transformer)
    dataloader = torch.utils.data.DataLoader(dset, batch_size=batchSize, shuffle=True,
                                             num_workers=num_workers, pin_memory=pin_memory)
    num_classes = 1
    return dataloader, num_classes


def DomainNet(datapath, batchSize, imgSize=64, num_workers=4, pin_memory=True,
           anomaly_frac=0, anomalypath=None, savepath='results', train=True):
    data_transformer = T.Compose([
        T.Resize(imgSize),
        T.RandomCrop(imgSize),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dset = datasets.ImageFolder(root=datapath, transform=data_transformer)
    dataloader = torch.utils.data.DataLoader(dset, batch_size=batchSize, shuffle=True,
                                             num_workers=num_workers, pin_memory=pin_memory)
    num_classes = 1
    return dataloader, num_classes



# TODO: Write anamalous data loaders
def get_anomaly_list(datapath):
    sample_list = []
    for root, _, fnames in sorted(os.walk(datapath)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            sample_list.append(path)
    return sample_list


# TODO: Imagenet data loader (use cached dataloader)

dataset_factory = {
    'cifar': cifar10,
    'cifar100': cifar100,
    'domainnet': DomainNet,
    'lsun': lsun,
    'celeba': CelebA,
    'celeba_attribute': CelebAWithAttributes,
    'celebahq': CelebAHQ
}
