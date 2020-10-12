import torch
import torchvision.transforms as transforms
import os.path as osp
from .folder import ImageFolder


def form_domainnet(config):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    nclasses = 345

    source_train_root = osp.join(config.dataroot, config.dataset, config.domain_src)
    target_root = osp.join(config.dataroot, config.dataset, config.domain_tgt)

    transform_source = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
    )

    transform_target = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
    )

    source_d = ImageFolder(root=source_train_root, transform=transform_source)
    target_d = ImageFolder(root=target_root, transform=transform_target)

    source_loader = torch.utils.data.DataLoader(source_d, batch_size=config.batchSize,
                                                shuffle=True, num_workers=4, pin_memory=True)
    target_loader = torch.utils.data.DataLoader(target_d, batch_size=config.batchSize,
                                                shuffle=True, num_workers=4, pin_memory=True)

    return source_loader, target_loader, nclasses
