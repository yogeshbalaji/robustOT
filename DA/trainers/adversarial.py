import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import torch.optim as optim
import datasets
import models
import utils
import losses
import cvxpy as cp
from pathlib import Path
import os.path as osp
import torchvision.models as torch_models
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


class AdversarialTrainer(object):
    def __init__(self, config):
        self.config = config
        self.device = 'cuda:0'

        # Create dataloader
        source_loader, target_loader, nclasses = datasets.form_visda_datasets(config=config, ignore_anomaly=True)
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.nclasses = nclasses

        # Create model
        self.netF, self.nemb = models.form_models(config)
        print(self.netF)
        self.netC = models.Classifier(self.nemb, self.nclasses, nlayers=1)
        utils.weights_init(self.netC)
        print(self.netC)
        self.netD = models.Classifier(self.nemb, 1, nlayers=3, use_spectral=True)
        utils.weights_init(self.netD)
        print(self.netD)

        self.netF = self.netF.to(self.device)
        self.netC = self.netC.to(self.device)
        self.netD = self.netD.to(self.device)

        self.netF = torch.nn.DataParallel(self.netF).cuda()
        self.netC = torch.nn.DataParallel(self.netC).cuda()
        self.netD = torch.nn.DataParallel(self.netD).cuda()

        # Create optimizer
        self.optimizerF = optim.SGD(self.netF.parameters(), lr=self.config.lr, momentum=config.momentum,
                                    weight_decay=0.0005)
        self.optimizerC = optim.SGD(self.netC.parameters(), lr=self.config.lrC, momentum=config.momentum,
                                    weight_decay=0.0005)
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.config.lrD, betas=(0.9, 0.999))

        self.lr_scheduler_F = utils.InvLR(self.optimizerF, gamma=0.0001, power=0.75)
        self.lr_scheduler_C = utils.InvLR(self.optimizerC, gamma=0.0001, power=0.75)

        # creating losses
        self.loss_fn = losses.loss_factory[config.loss]
        self.entropy_criterion = losses.EntropyLoss()

        self.best_acc = 0

        # restoring checkpoint
        print('Restoring checkpoint ...')
        try:
            ckpt_path = os.path.join(config.logdir, 'model_state.pth')
            self.restore_state(ckpt_path)
        except:
            # If loading failed, begin from scratch
            print('Checkpoint not found. Training from scratch ...')
            self.itr = 0
            self.epoch = 0

    def save_state(self):
        model_state = {}
        model_state['epoch'] = self.epoch
        model_state['itr'] = self.itr
        self.netF.eval()
        self.netD.eval()
        self.netC.eval()
        model_state['netD'] = self.netD.state_dict()
        model_state['netF'] = self.netF.state_dict()
        model_state['netC'] = self.netF.state_dict()
        model_state['optimizerD'] = self.optimizerD.state_dict()
        model_state['optimizerF'] = self.optimizerF.state_dict()
        model_state['optimizerC'] = self.optimizerC.state_dict()
        model_state['best_acc'] = self.best_acc

        torch.save(model_state, osp.join(self.config.logdir, 'model_state.pth'))

    def restore_state(self, pth):
        print('Restoring state ...')
        model_state = torch.load(pth)
        self.epoch = model_state['epoch']
        self.itr = model_state['itr']
        self.best_acc = model_state['best_acc']
        self.netD.load_state_dict(model_state['netD'])
        self.netF.load_state_dict(model_state['netF'])
        self.netC.load_state_dict(model_state['netC'])
        self.optimizerD.load_state_dict(model_state['optimizerD'])
        self.optimizerF.load_state_dict(model_state['optimizerF'])
        self.optimizerC.load_state_dict(model_state['optimizerC'])

    def zero_grad_all(self):
        self.netF.zero_grad()
        self.netC.zero_grad()
        self.netD.zero_grad()
        self.optimizerF.zero_grad()
        self.optimizerC.zero_grad()
        self.optimizerD.zero_grad()

    def log(self, message):
        print(message)
        message = message + '\n'
        f = open("{}/log.txt".format(self.config.logdir), "a+")
        f.write(message)
        f.close()

    def test(self):
        self.netF.eval()
        self.netC.eval()

        correct = 0
        size = 0
        num_class = self.nclasses
        output_all = np.zeros((0, num_class))
        confusion_matrix = torch.zeros(num_class, num_class)

        with torch.no_grad():
            for batch_idx, data_t in enumerate(self.target_loader):
                imgs, labels, _ = data_t
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

        print('\nTest set: Accuracy: {}/{} C ({:.0f}%)\n'.format(correct, size, 100. * float(correct) / size))
        mean_class_acc = torch.diagonal(confusion_matrix) / torch.sum(confusion_matrix, dim=1)
        mean_class_acc = mean_class_acc * 100.0

        print('Classwise accuracy')
        print(mean_class_acc)

        mean_class_acc = torch.mean(mean_class_acc)
        net_class_acc = 100. * float(correct) / size
        return mean_class_acc, net_class_acc

    def train(self):
        print('Start training from iter {}'.format(self.itr))
        end_flag = 0

        while True:
            self.epoch += 1
            if end_flag == 1:
                break

            for i, (data_s, data_t) in enumerate(zip(self.source_loader, self.target_loader)):
                self.itr += 1
                if self.itr > self.config.num_iters:
                    print('Training complete')
                    end_flag = 1
                    break

                self.netF.train()
                self.netC.train()
                self.netD.train()

                inp_s, lab_s, indices_src = data_s
                inp_s, lab_s = inp_s.to(self.device), lab_s.to(self.device)

                inp_t, lab_t, indices_tgt = data_t
                inp_t = inp_t.to(self.device)

                self.zero_grad_all()

                feat_s = self.netF(inp_s, dom_id=0)
                feat_t = self.netF(inp_t, dom_id=1)

                # adversarial loss
                disc_logits_s = self.netD(feat_s)
                disc_logits_t = self.netD(feat_t)
                weights = torch.ones(feat_t.size(0), 1).to(self.device)

                errD = self.loss_fn(disc_logits_s, disc_logits_t, weights)
                errD.backward(retain_graph=True)

                if self.config.regularization == 'gradient_penalty':
                    gp = losses.gradient_penalty(self.netD, feat_s, feat_t, self.config.gp_lamb,
                                                 device=self.device)
                    gp.backward()
                self.optimizerD.step()

                self.optimizerF.zero_grad()
                self.optimizerC.zero_grad()

                if self.itr % self.config.disc_iters == 0:
                    errG = -1 * self.loss_fn(disc_logits_s, disc_logits_t, weights)
                    errG.backward(retain_graph=True)

                logits_t = self.netC(feat_t)
                ent_loss = self.config.ent_weight * self.entropy_criterion(logits_t)
                ent_loss.backward()

                # VAT loss
                if self.config.vat_weight > 0:
                    if i == 0:
                        vat_loss = self.config.vat_weight * losses.vat_criterion(self.netF, self.netC, inp_t, debug=True)
                    else:
                        vat_loss = self.config.vat_weight * losses.vat_criterion(self.netF, self.netC, inp_t)
                    vat_loss.backward()

                # Classification loss
                logits = self.netC(feat_s)
                lossC = F.cross_entropy(logits, lab_s)
                lossC.backward()

                self.optimizerF.step()
                self.optimizerC.step()

                self.lr_scheduler_F.step()
                self.lr_scheduler_C.step()

                lr = self.optimizerF.param_groups[0]['lr']

                if self.itr % self.config.log_interval == 0:
                    log_train = 'Train iter: {}, Epoch: {}, lr{} \t Loss Classification: {:.6f} ' \
                                'Method {}'.format(self.itr, self.epoch, lr, lossC.item(), self.config.method)
                    self.log(log_train)

                if self.itr % self.config.save_interval == 0:
                    mean_class_acc, net_class_acc = self.test()
                    if mean_class_acc > self.best_acc:
                        self.best_acc = mean_class_acc
                    msg = 'Mean class acc: {}, Net class acc: {}'.format(mean_class_acc, net_class_acc)
                    self.log(msg)
                    msg = 'Best class acc: {}'.format(self.best_acc)
                    self.log(msg)

                    print('Saving model')
                    self.save_state()
                    self.netF.train()
                    self.netC.train()
                    self.netD.train()
