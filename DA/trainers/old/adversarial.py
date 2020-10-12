import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import datasets
import models
import utils
import losses


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
            ckpt_data = torch.load(os.path.join(config.logdir, 'checkpoint.pth'))
            self.start_iter = ckpt_data['iter']
            self.netF.load_state_dict(ckpt_data['F_dict'])
            self.netC.load_state_dict(ckpt_data['C_dict'])
            self.netD.load_state_dict(ckpt_data['D_dict'])
        except:
            # If loading failed, begin from scratch
            print('Checkpoint not found. Training from scratch ...')
            self.start_iter = 0

    def zero_grad_all(self):
        self.optimizerF.zero_grad()
        self.optimizerC.zero_grad()

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
        ncls = num_class
        confusion_matrix = torch.zeros(ncls, ncls)

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
        self.netF.train()
        self.netC.train()

        print('Start training from iter {}'.format(self.start_iter))
        num_iter = self.start_iter
        num_epoch = 0
        end_flag = 0

        while True:
            num_epoch += 1
            if end_flag == 1:
                break

            for i, (data_s, data_t) in enumerate(zip(self.source_loader, self.target_loader)):
                num_iter += 1
                if num_iter > self.config.num_iters:
                    print('Training complete')
                    end_flag = 1
                    break

                inp_s, lab_s, _ = data_s
                inp_s, lab_s = inp_s.to(self.device), lab_s.to(self.device)

                inp_t, _, _ = data_t
                inp_t = inp_t.to(self.device)

                self.zero_grad_all()

                # dummy forward pass
                feat_s = self.netF(inp_s)
                feat_t = self.netF(inp_t)

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

                if num_iter % self.config.disc_iters == 0:
                    errG = self.loss_fn(disc_logits_s, disc_logits_t, weights) * -1
                    errG.backward(retain_graph=True)

                # Entropy loss
                logits_t = self.netC(feat_t)
                ent_loss = self.config.ent_weight * self.entropy_criterion(logits_t)
                ent_loss.backward()

                # VAT loss
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

                if num_iter % self.config.log_interval == 0:
                    log_train = 'Train iter: {}, Epoch: {}, lr{} \t Loss Classification: {:.6f} ' \
                                'Method {}'.format(num_iter, num_epoch, lr, lossC.item(), self.config.method)
                    self.log(log_train)

                if num_iter % self.config.save_interval == 0:
                    mean_class_acc, net_class_acc = self.test()
                    if self.best_acc < mean_class_acc:
                        self.best_acc = mean_class_acc
                    msg = 'Mean class acc: {}, Net class acc: {}'.format(mean_class_acc, net_class_acc)
                    self.log(msg)
                    msg = 'Best acc: {}'.format(self.best_acc)
                    self.log(msg)

                    print('Saving model')
                    ckpt_data = dict()
                    ckpt_data['iter'] = num_iter
                    ckpt_data['F_dict'] = self.netF.state_dict()
                    ckpt_data['C_dict'] = self.netC.state_dict()
                    torch.save(ckpt_data, os.path.join(self.config.logdir, 'checkpoint.pth'))
                    self.netF.train()
                    self.netC.train()
