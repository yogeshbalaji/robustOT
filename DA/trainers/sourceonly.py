import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import datasets
import models
import utils


class SourceonlyTrainer(object):
    def __init__(self, config):
        self.config = config

        # Create dataloader
        source_loader, target_loader, nclasses = datasets.form_visda_datasets(config=config, ignore_anomaly=False)
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.nclasses = nclasses

        # Create model
        self.netF, self.nemb = models.form_models(config)
        print(self.netF)
        self.netC = models.Classifier(self.nemb, self.nclasses, nlayers=1)
        utils.weights_init(self.netC)
        print(self.netC)

        if self.config.exp == 'openset':
            self.ano_class_id = self.source_loader.dataset.class_to_idx[self.config.anomaly_class]

        self.netF = torch.nn.DataParallel(self.netF).cuda()
        self.netC = torch.nn.DataParallel(self.netC).cuda()

        # Create optimizer
        self.optimizerF = optim.SGD(self.netF.parameters(), lr=self.config.lr, momentum=config.momentum,
                                    weight_decay=0.0005)
        self.optimizerC = optim.SGD(self.netC.parameters(), lr=self.config.lrC, momentum=config.momentum,
                                    weight_decay=0.0005)
        self.lr_scheduler_F = optim.lr_scheduler.StepLR(self.optimizerF, step_size=7000,
                                                        gamma=0.1)
        self.lr_scheduler_C = optim.lr_scheduler.StepLR(self.optimizerC, step_size=7000,
                                                        gamma=0.1)

        # restoring checkpoint
        print('Restoring checkpoint ...')
        try:
            ckpt_data = torch.load(os.path.join(config.logdir, 'checkpoint.pth'))
            self.start_iter = ckpt_data['iter']
            self.netF.load_state_dict(ckpt_data['F_dict'])
            self.netC.load_state_dict(ckpt_data['C_dict'])
        except:
            # If loading failed, begin from scratch
            print('Checkpoint not found. Training from scratch ...')
            self.start_iter = 0

        # Other vars
        self.criterion = nn.CrossEntropyLoss().cuda()

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

        test_loss = 0
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
                test_loss += self.criterion(logits, labels) / len(self.target_loader)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} C ({:.0f}%)\n'.format(test_loss, correct, size,
                                                                                       100. * float(correct) / size))
        mean_class_acc = torch.diagonal(confusion_matrix) / torch.sum(confusion_matrix, dim=1)
        mean_class_acc = mean_class_acc * 100.0

        print('Classwise accuracy')
        print(mean_class_acc)

        if self.config.exp == 'openset':
            OS = torch.mean(mean_class_acc)
            OS_star_cls = np.array([mean_class_acc[i] for i in range(len(mean_class_acc)) if i != self.ano_class_id])
            print(OS_star_cls)
            OS_star = np.mean(OS_star_cls)
            print('OS: {}'.format(OS))
            print('OS star: {}'.format(OS_star))

            return test_loss.data, OS, OS_star
        else:
            mean_class_acc = torch.mean(mean_class_acc)
            net_class_acc = 100. * float(correct) / size
            return test_loss.data, mean_class_acc, net_class_acc

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
                inp_s, lab_s = inp_s.cuda(), lab_s.cuda()

                inp_t, _, _ = data_t
                inp_t = inp_t.cuda()

                self.zero_grad_all()

                # dummy forward pass
                logit_t = self.netF(inp_t)
                logits = self.netC(self.netF(inp_s))
                loss = self.criterion(logits, lab_s)
                loss.backward()
                self.optimizerC.step()
                self.optimizerF.step()

                self.lr_scheduler_F.step()
                self.lr_scheduler_C.step()

                lr = self.optimizerF.param_groups[0]['lr']

                if num_iter % self.config.log_interval == 0:
                    log_train = 'Train iter: {}, Epoch: {}, lr{} \t Loss Classification: {:.6f} ' \
                                'Method {}'.format(num_iter, num_epoch, lr, loss.item(), self.config.method)
                    self.log(log_train)

                if num_iter % self.config.save_interval == 0:
                    if self.config.exp == 'openset':
                        test_loss, mean_class_acc, net_class_acc = self.test()
                        msg = 'Test loss: {}, OS: {}, OS star: {}'.format(test_loss, mean_class_acc, net_class_acc)
                        self.log(msg)
                    else:
                        test_loss, mean_class_acc, net_class_acc = self.test()
                        msg = 'Test loss: {}, Mean class acc: {}, Net class acc: {}'.format(test_loss, mean_class_acc,
                                                                                            net_class_acc)
                        self.log(msg)

                    print('Saving model')
                    ckpt_data = dict()
                    ckpt_data['iter'] = num_iter
                    ckpt_data['F_dict'] = self.netF.state_dict()
                    ckpt_data['C_dict'] = self.netC.state_dict()
                    torch.save(ckpt_data, os.path.join(self.config.logdir, 'checkpoint.pth'))
                    self.netF.train()
                    self.netC.train()
