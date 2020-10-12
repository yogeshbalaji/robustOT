import torch
import torchvision.transforms as transforms
import os.path as osp
from .folder import ImageFolder


def form_visda18(config, ignore_anomaly=False, ano_type=1):

    ano_type_dict = {
        1: 'validation',
        2: 'validation_1_1',
        3: 'validation_1_0.2'
    }
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if ignore_anomaly:
        nclasses = 12
    else:
        nclasses = 13

    source_root = osp.join(config.dataroot, config.dataset, 'train')
    target_root = osp.join(config.dataroot, config.dataset, ano_type_dict[ano_type])
    print('Reading target from {}'.format(target_root))

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

    source_d = ImageFolder(root=source_root, transform=transform_source, anomaly_class=[config.anomaly_class],
                           ignore_anomaly=ignore_anomaly)
    target_d = ImageFolder(root=target_root, transform=transform_target, anomaly_class=[config.anomaly_class])

    source_loader = torch.utils.data.DataLoader(source_d, batch_size=config.batchSize,
                                                shuffle=True, num_workers=4)
    target_loader = torch.utils.data.DataLoader(target_d, batch_size=config.batchSize,
                                                shuffle=True, num_workers=4)
    return source_loader, target_loader, nclasses
