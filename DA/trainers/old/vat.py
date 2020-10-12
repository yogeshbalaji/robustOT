import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datasets
import models
import utils
import contextlib
import torchvision.utils as vutils
from pathlib import Path

@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


class VATPerturbationGenerator(nn.Module):

    def __init__(self, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATPerturbationGenerator, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, modelF, modelC, x):
        with torch.no_grad():
            pred = F.softmax(modelC(modelF(x)), dim=1)

        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(modelF):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = modelC(modelF(x + self.xi * d))
                logp_hat = F.log_softmax(pred_hat, dim=1)
                adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                modelF.zero_grad()
                modelC.zero_grad()

            # calc LDS
            r_adv = d * self.eps

        x_pert = x + r_adv.detach()
        return x_pert


class EntropyLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits):
        p = F.softmax(logits, dim=1)
        elementwise_entropy = -p * F.log_softmax(logits, dim=1)
        if self.reduction == 'none':
            return elementwise_entropy

        sum_entropy = torch.sum(elementwise_entropy, dim=1)
        if self.reduction == 'sum':
            return sum_entropy

        return torch.mean(sum_entropy)


class VATTrainer(object):
    def __init__(self, args):
        self.args = args

        # Create dataloader
        source_train_loader, source_val_loader, target_loader, nclasses = datasets.form_visda_datasets(config=args)
        self.source_train_loader = source_train_loader
        self.source_val_loader = source_val_loader
        self.target_loader = target_loader
        self.nclasses = nclasses

        # Create model
        if args.model == 'resnet18':
            self.netF = models.resnet18(pretrained=True)
            self.nemb = 512
        elif args.model == 'resnet34':
            self.netF = models.resnet34(pretrained=True)
            self.nemb = 512
        elif args.model == 'resnet50':
            self.netF = models.resnet50(pretrained=True)
            self.nemb = 2048
        elif args.model == 'resnet101':
            self.netF = models.resnet101(pretrained=True)
            self.nemb = 2048
        elif args.model == 'resnet152':
            self.netF = models.resnet152(pretrained=True)
            self.nemb = 2048
        else:
            raise ValueError('Model cannot be recognized.')

        print(self.netF)
        self.netC = models.Classifier(self.nemb, self.nclasses, nlayers=1)
        utils.weights_init(self.netC)
        print(self.netC)

        self.netF = torch.nn.DataParallel(self.netF).cuda()
        self.netC = torch.nn.DataParallel(self.netC).cuda()

        # Create optimizer
        self.optimizerF = optim.SGD(self.netF.parameters(), lr=self.args.lr, momentum=args.momentum,
                                    weight_decay=0.0005)
        self.optimizerC = optim.SGD(self.netC.parameters(), lr=self.args.lrC, momentum=args.momentum,
                                    weight_decay=0.0005)
        self.lr_scheduler_F = optim.lr_scheduler.StepLR(self.optimizerF, step_size=7000,
                                                        gamma=0.1)
        self.lr_scheduler_C = optim.lr_scheduler.StepLR(self.optimizerC, step_size=7000,
                                                        gamma=0.1)

        # restoring checkpoint
        print('Restoring checkpoint ...')
        try:
            ckpt_data = torch.load(os.path.join(args.save_path, 'checkpoint.pth'))
            self.start_iter = ckpt_data['iter']
            self.netF.load_state_dict(ckpt_data['F_dict'])
            self.netC.load_state_dict(ckpt_data['C_dict'])
        except:
            # If loading failed, begin from scratch
            print('Checkpoint not found. Training from scratch ...')
            self.start_iter = 0

        # Other vars
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.vat_pert_gen = VATPerturbationGenerator(xi=10.0, eps=1.0, ip=1)
        self.entropy_criterion = EntropyLoss()

    def zero_grad_all(self):
        self.optimizerF.zero_grad()
        self.optimizerC.zero_grad()

    def vat_criterion(self, modelF, modelC, inp, debug=False):
        pert = self.vat_pert_gen(modelF, modelC, inp)

        with _disable_tracking_bn_stats(modelF):
            with torch.no_grad():
                pred = modelC(modelF(inp))

            pred_hat = modelC(modelF(pert))
            logp_hat = F.log_softmax(pred_hat, dim=1)
            lds = F.kl_div(logp_hat, pred, reduction='batchmean')

        if debug:
            Path('{}/debug'.format(self.args.save_path)).mkdir(exist_ok=True)
            print('Perturbation norm: {}'.format(torch.norm((pert - inp)[0])))
            vutils.save_image(inp, '{}/debug/orig.png'.format(self.args.save_path), normalize=True)
            vutils.save_image(pert, '{}/debug/perturbed.png'.format(self.args.save_path), normalize=True)

        return lds

    def log(self, message):
        print(message)
        message = message + '\n'
        f = open("{}/log.txt".format(self.args.save_path), "a+")
        f.write(message)
        f.close()

    def test(self, validation=False):
        self.netF.eval()
        self.netC.eval()

        test_loss = 0
        correct = 0
        size = 0
        num_class = self.nclasses
        output_all = np.zeros((0, num_class))
        confusion_matrix = torch.zeros(num_class, num_class)

        if validation:
            loader = self.source_val_loader
        else:
            loader = self.target_loader

        with torch.no_grad():
            for batch_idx, data_t in enumerate(loader):
                imgs, labels = data_t
                imgs = imgs.cuda()
                labels = labels.cuda()

                feat = self.netF(imgs)
                logits = self.netC(feat)
                output_all = np.r_[output_all, logits.data.cpu().numpy()]
                size += imgs.size(0)
                pred = logits.data.max(1)[1]  # get the index of the max log-probability
                for t, p in zip(labels.view(-1), pred.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                correct += pred.eq(labels.data).cpu().sum()
                test_loss += self.criterion(logits, labels) / len(loader)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} C ({:.0f}%)\n'.format(test_loss, correct, size,
                                                                                       100. * (float(correct) / size)))
        mean_class_acc = torch.diagonal(confusion_matrix).float() / torch.sum(confusion_matrix, dim=1)
        mean_class_acc = mean_class_acc * 100.0
        print('Classwise accuracy')
        print(mean_class_acc)
        mean_class_acc = torch.mean(mean_class_acc)
        net_class_acc = 100. * float(correct) / size
        return test_loss.data, mean_class_acc, net_class_acc

    def train(self):
        self.netF.train()
        self.netC.train()

        print('Start training from iter {}'.format(self.start_iter))
        num_iter = self.start_iter
        epoch_num = 0
        end_flag = False
        while True:
            epoch_num += 1
            correct_train = 0
            total_train = 0
            if end_flag:
                break
            for i, (data_s, data_t) in enumerate(zip(self.source_train_loader, self.target_loader)):
                num_iter += 1
                if num_iter > self.args.num_iters:
                    print('Training complete')
                    end_flag = True
                    break

                inp_s, lab_s = data_s
                inp_s, lab_s = inp_s.cuda(), lab_s.cuda()

                inp_t, _ = data_t
                inp_t = inp_t.cuda()

                self.zero_grad_all()

                # Regularization
                target_logits = self.netC(self.netF(inp_t))
                ent_loss = self.args.ent_weight * self.entropy_criterion(target_logits)
                if i == 0:
                    vat_loss = self.args.vat_weight * self.vat_criterion(self.netF, self.netC, inp_t, debug=True)
                else:
                    vat_loss = self.args.vat_weight * self.vat_criterion(self.netF, self.netC, inp_t)
                reg = ent_loss + vat_loss
                reg.backward()

                # Cross entropy loss on source
                logits = self.netC(self.netF(inp_s))
                loss_cls = self.criterion(logits, lab_s)
                loss_cls.backward()

                _, preds_train = torch.max(logits, dim=1)
                correct_train += (preds_train == lab_s).sum()
                total_train += lab_s.size(0)

                self.optimizerC.step()
                self.optimizerF.step()

                self.lr_scheduler_F.step()
                self.lr_scheduler_C.step()

                lr = self.optimizerF.param_groups[0]['lr']

                if num_iter % self.args.log_interval == 0:
                    acc_train = (float(correct_train) / total_train) * 100.0
                    log_train = 'Train iter: {}, Epoch: {}, lr{} \t Loss Classification: {:.6f} ' \
                                'Train acc:{:.4f}, Reg {:.4f}, Method {}'.format(num_iter, epoch_num, lr,
                                                                                 loss_cls.item(), acc_train,
                                                                                 reg.item(), self.args.method)
                    self.log(log_train)

                if num_iter % self.args.save_interval == 0:
                    test_loss, mean_class_acc, net_class_acc = self.test(validation=True)
                    msg = 'Source validation loss: {}, Mean class acc: {}, Net class acc: {}'.format(test_loss,
                                                                                                     mean_class_acc,
                                                                                                     net_class_acc)
                    self.log(msg)

                    test_loss, mean_class_acc, net_class_acc = self.test(validation=False)
                    msg = 'Target validation loss: {}, Mean class acc: {}, Net class acc: {}'.format(test_loss,
                                                                                                     mean_class_acc,
                                                                                                     net_class_acc)
                    self.log(msg)

                    print('Saving model')
                    ckpt_data = dict()
                    ckpt_data['iter'] = num_iter
                    ckpt_data['F_dict'] = self.netF.state_dict()
                    ckpt_data['C_dict'] = self.netC.state_dict()
                    torch.save(ckpt_data, os.path.join(self.args.save_path, 'checkpoint.pth'))

                    self.netF.train()
                    self.netC.train()


