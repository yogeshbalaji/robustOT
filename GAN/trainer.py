import torch.optim as optim
import torch
import torch.nn as nn
import utils
from datasets import dataset_factory
import models
import losses
import os.path as osp
from pathlib import Path
import torchvision.utils as vutils
import math
import inception
import numpy as np
import torch.nn.functional as F
import copy
import cvxpy as cp
import os


class BaseTrainer(object):
    def __init__(self, config):
        self.config = config
        self.device = 'cuda:0'

        dset_fn = dataset_factory[config.dataset]

        # Creating dataloader
        if config.dataset == 'celeba_attribute':

            # Values should contain a tuple of (fraction, enable_flag)
            attribute_list = [
                ('Male', 1, config.anomaly_frac),
                ('Male', -1, 1 - config.anomaly_frac)
            ]
            self.dataloader, self.num_classes = dset_fn(config.dataroot, config.batchSize,
                                                        imgSize=config.imageSize,
                                                        input_attribute_list=attribute_list,
                                                        anomaly_frac=config.anomaly_frac,
                                                        anomalypath=config.anopath,
                                                        savepath=config.logdir,
                                                        train=True)
            self.testloader, _ = dset_fn(config.dataroot, 32,
                                         imgSize=config.imageSize,
                                         input_attribute_list=attribute_list,
                                         anomaly_frac=config.anomaly_frac,
                                         anomalypath=config.anopath,
                                         train=False)
        else:
            self.dataloader, self.num_classes = dset_fn(config.dataroot, config.batchSize, imgSize=config.imageSize,
                                                        anomaly_frac=config.anomaly_frac, anomalypath=config.anopath,
                                                        savepath=config.logdir, train=True)
            self.testloader, _ = dset_fn(config.dataroot, 32, imgSize=config.imageSize,
                                         anomaly_frac=config.anomaly_frac, anomalypath=config.anopath, train=False)
        config.__dict__['num_classes'] = self.num_classes

        if self.num_classes == 1:
            assert not config.conditional
        # Creating models
        gen_model_fn = models.generator_factory[config.netG]
        disc_model_fn = models.discriminator_factory[config.netD]
        self.netG = gen_model_fn(config)
        self.netD = disc_model_fn(config)
        self.netG = self.netG.to(self.device)
        self.netD = self.netD.to(self.device)
        print(self.netD)
        print(self.netG)
        if self.config.ngpu > 1:
            self.netG = nn.DataParallel(self.netG)
            self.netD = nn.DataParallel(self.netD)

        print(self.netG)
        print(self.netD)

        # Creating optimizer
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=config.lrD, betas=(config.beta1D, config.beta2D))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=config.lrG, betas=(config.beta1G, config.beta2G))

        num_iters = (len(self.dataloader.dataset) / config.batchSize) * config.nepochs
        print('Running for {} discriminator iterations'.format(num_iters))
        if config.lrdecay:
            self.schedulerD = utils.LinearLR(self.optimizerD, num_iters)
            self.schedulerG = utils.LinearLR(self.optimizerG, num_iters)

        # Creating loss functions
        [self.disc_loss_fn, self.gen_loss_fn] = losses.loss_factory[config.loss]
        self.aux_loss_fn = losses.aux_loss

        self.epoch = 0
        self.itr = 0
        Path(self.config.logdir).mkdir(exist_ok=True, parents=True)

        if config.use_ema:
            self.ema = utils.ema(self.netG)

        self.log_path = osp.join(self.config.logdir, 'log.txt')
        if not osp.exists(self.log_path):
            fh = open(self.log_path, 'w')
            fh.write('Logging {} GAN training\n'.format(self.config.dataset))
            fh.close()

        self.inception_evaluator = inception.Evaluator(config, self.testloader)
        self.prev_gen_loss = 0
        self.best_is = 0
        self.best_fid = 100000
        self.best_is_std = 0
        self.best_intra_fid = 1000000
        self.eps = 0.0001

        # Weights
        self.rho = self.config.rho
        self.weight_update_flag = self.config.weight_update
        self.weight_update_type = self.config.weight_update_type
        if self.weight_update_flag:
            if self.weight_update_type == 'discrete':
                self.num_datapoints = len(self.dataloader.dataset)
                self.weight_vector = torch.FloatTensor(self.num_datapoints, ).fill_(1).to(self.device)
                self.disc_vector = torch.FloatTensor(self.num_datapoints, ).fill_(1).to(self.device)
                self.disc_vector_cur = torch.FloatTensor(self.num_datapoints, ).fill_(1).to(self.device)
            else:
                weight_model_fn = models.weight_factory[config.netD]
                self.netW = weight_model_fn(config).to(self.device)
                if self.config.ngpu > 1:
                    self.netW = nn.DataParallel(self.netW)
                self.optimizerW = optim.Adam(self.netW.parameters(), lr=config.lrD,
                                             betas=(config.beta1D, config.beta2D))
                self.weight_loss_fn = losses.loss_factory_weights[config.loss]

        # Code for restoring models from checkpoint
        if config.restore != '':
            self.restore_state(config.restore)

        # Checking if state exists (in case of preemption)
        if os.path.exists(osp.join(self.config.logdir, 'model_state.pth')):
            self.restore_state(osp.join(self.config.logdir, 'model_state.pth'))

    def log(self, message):
        print(message)
        fh = open(self.log_path, 'a+')
        fh.write(message + '\n')
        fh.close()

    def save_state(self, savename=None):
        model_state = {}
        model_state['epoch'] = self.epoch
        model_state['itr'] = self.itr
        if savename is None:
            savename = 'model_state.pth'
        self.netD.eval()
        self.netG.eval()
        model_state['netD'] = self.netD.state_dict()
        model_state['netG'] = self.netG.state_dict()
        model_state['optimizerD'] = self.optimizerD.state_dict()
        model_state['optimizerG'] = self.optimizerG.state_dict()
        model_state['best_is'] = self.best_is
        model_state['best_is_std'] = self.best_is_std
        model_state['best_fid'] = self.best_fid
        model_state['best_intra_fid'] = self.best_intra_fid
        if self.weight_update_flag:
            if self.weight_update_type == 'discrete':
                model_state['weight_vector'] = self.weight_vector.cpu()
            else:
                model_state['netW'] = self.netW.state_dict()
                model_state['optimizerW'] = self.optimizerW.state_dict()
        torch.save(model_state, osp.join(self.config.logdir, savename))

    def restore_state(self, pth):
        print('Restoring state ...')
        model_state = torch.load(pth)
        self.epoch = model_state['epoch']
        self.itr = model_state['itr']
        self.best_is = model_state['best_is']
        self.best_is_std = model_state['best_is_std']
        self.best_fid = model_state['best_fid']
        self.best_intra_fid = model_state['best_intra_fid']
        self.netD.load_state_dict(model_state['netD'])
        self.netG.load_state_dict(model_state['netG'])
        self.optimizerD.load_state_dict(model_state['optimizerD'])
        self.optimizerG.load_state_dict(model_state['optimizerG'])
        if self.weight_update_flag:
            if self.weight_update_type == 'discrete':
                self.weight_vector = model_state['weight_vector']
                self.weight_vector = self.weight_vector.to(self.device)
            else:
                self.netW.load_state_dict(model_state['netW'])
                self.optimizerW.load_state_dict(model_state['optimizerW'])


class GAN(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)

    def gan_updates(self, real_data, real_labels, real_indices):

        self.optimizerD.zero_grad()

        batch_size = real_data.size(0)
        noise = utils.sample_normal(batch_size, self.config.nz, device=self.device)
        if self.config.conditional:
            fake_labels = utils.sample_cats(batch_size, self.num_classes, device=self.device)
        else:
            real_labels = None
            fake_labels = None
        
        fake_data = self.netG(noise, fake_labels)

        # Discriminator updates
        outD_real = self.netD(real_data, real_labels)
        outD_fake = self.netD(fake_data.detach(), fake_labels)
        if self.weight_update_flag and self.weight_update_type == 'discrete':
            self.disc_vector_cur[real_indices] = torch.squeeze(outD_real)

        if self.weight_update_flag:
            if self.weight_update_type == 'discrete':
                real_weights = self.weight_vector[real_indices].view(-1, 1)
            else:
                real_weights = self.netW(real_data, real_labels) + self.eps
                real_weights = (real_weights / real_weights.sum()) * self.config.batchSize
        else:
            real_weights = torch.ones(real_data.size(0), 1).to(self.device)

        if self.config.conditioning == 'acgan':

            outD_real_cls = outD_real[1]
            outD_real = outD_real[0]
            outD_fake_cls = outD_fake[1]
            outD_fake = outD_fake[0]
            aux_loss_real = self.aux_loss_fn(outD_real_cls, real_labels)
            aux_loss_fake = self.aux_loss_fn(outD_fake_cls, fake_labels)

        errD_real, errD_fake = self.disc_loss_fn(outD_real, outD_fake, real_weights)
        if self.config.conditioning == 'acgan':
            errD_real = errD_real + aux_loss_real
            errD_fake = errD_fake + aux_loss_fake

        errD_real.backward()
        errD_fake.backward()

        if self.config.regularization == 'gradient_penalty':
            if self.config.conditional:
                fake_data_consistent = self.netG(noise, real_labels)
                gp = losses.gradient_penalty(self.netD, real_data, fake_data_consistent, self.config.gp_lamb,
                                             device=self.device, labels=real_labels)
            else:
                gp = losses.gradient_penalty(self.netD, real_data, fake_data, self.config.gp_lamb, device=self.device)
            gp.backward()
        if self.config.regularization == 'ortho':
            losses.orthogonal_regularization(self.netD, self.config.ortho_strength)

        self.optimizerD.step()
        if self.config.lrdecay:
            self.schedulerD.step()
            self.schedulerG.step()

        disc_loss = errD_real.item() + errD_fake.item()

        # Generator updates
        if self.itr % self.config.disc_iters == 0:
            self.optimizerG.zero_grad()
            outD = self.netD(fake_data, fake_labels)

            if self.config.conditioning == 'acgan':
                outD_cls = outD[1]
                outD = outD[0]
                aux_loss = self.aux_loss_fn(outD_cls, fake_labels)

            errG = self.gen_loss_fn(outD)
            if self.config.conditioning == 'acgan':
                errG = errG + aux_loss
            errG.backward()

            self.optimizerG.step()
            gen_loss = errG.item()
            self.prev_gen_loss = gen_loss
        else:
            gen_loss = self.prev_gen_loss

        return disc_loss, gen_loss

    def weight_updates(self, real_data, real_labels, vis=True):
        # Module for updating weights

        if self.weight_update_type == 'discrete':
            m = self.num_datapoints

            # Populating discriminator outputs
            disc_arr = torch.zeros(m, )
            with torch.no_grad():
                for data in self.dataloader:
                    inp, labels, indices = data
                    inp = inp.to(self.device)

                    disc_out = self.netD(inp, real_labels)
                    disc_out = disc_out.view(-1)
                    disc_out = disc_out.cpu()
                    disc_arr[indices] = disc_out

            if self.config.disc_momentum > 0:
                disc_arr = disc_arr * (1 - self.config.disc_momentum) + self.config.disc_momentum * self.disc_vector.cpu()
            disc_arr = disc_arr.detach().numpy()

            # Solving convex optimization problem
            # Note: we are using normalized weights
            weight_arr = cp.Variable((self.num_datapoints,))
            ones = np.ones(m)

            soc_const = cp.Constant(np.sqrt(2 * self.config.rho * m))
            constraints = [cp.SOC(soc_const, (weight_arr - ones)),
                           cp.matmul(weight_arr.T, ones) == m, weight_arr >= 0]
            objective = cp.Minimize(cp.matmul(weight_arr.T, disc_arr))

            prob = cp.Problem(objective, constraints)
            result = prob.solve(solver='SCS')

            weight_res = weight_arr.value
            weight_res = torch.from_numpy(weight_res)

            self.weight_vector.copy_(weight_res)

        else:
            self.optimizerW.zero_grad()
            real_weights = self.netW(real_data, real_labels) + self.eps
            real_weights = (real_weights / real_weights.sum()) * self.config.batchSize
            real_logits = self.netD(real_data, real_labels)

            # Chi-squared
            soft_constraint = 100 * F.relu(torch.mean(0.5 * ((real_weights - 1) ** 2)) - self.config.rho)

            # Total variation
            # soft_constraint = 1000 * F.relu(torch.mean(0.5 * torch.abs(real_weights - 1)) - self.config.rho)

            loss_weights = self.weight_loss_fn(real_logits, real_weights) + soft_constraint
            loss_weights.backward()
            self.optimizerW.step()

            if vis:
                img_path = osp.join(self.config.logdir, 'samples')
                Path(img_path).mkdir(parents=True, exist_ok=True)
                real_weights_sorted, indices = torch.sort(real_weights.view(-1))
                print('Weights')
                print(real_weights_sorted)
                print('Soft constraint: {}'.format(soft_constraint.item()))
                vutils.save_image(real_data[indices, ::] * 0.5 + 0.5, '{}/real_vis.png'.format(img_path))
                torch.save(real_weights_sorted, '{}/weights.pth'.format(img_path))

    def save_samples(self):
        self.netG.eval()
        img_path = osp.join(self.config.logdir, 'samples')
        Path(img_path).mkdir(parents=True, exist_ok=True)

        def sample_and_save(prefix='normal'):
            sample_bs = 100
            bs = 100
            if self.config.dataset == 'cifar100':
                bs = 1000
            z = utils.sample_normal(bs, self.config.nz, device=self.device)
            if self.config.conditional:
                num_rep = int(math.ceil(bs / self.num_classes))
                y = [[i] * num_rep for i in range(self.num_classes)]
                y = np.hstack(y)
                y = torch.from_numpy(y).long()
                y = y.to(self.device)
            else:
                y = None

            gen_list = []
            with torch.no_grad():
                for i in range(int(bs / sample_bs)):
                    z_cur = z[i * sample_bs: (i + 1) * sample_bs]
                    if self.config.conditional:
                        y_cur = y[i * sample_bs: (i + 1) * sample_bs]
                    else:
                        y_cur = None
                    gen = self.netG(z_cur, y_cur)
                    gen_list.append(gen)
            gen = torch.cat(gen_list, dim=0)
            vutils.save_image(gen * 0.5 + 0.5, '{}/{}_{}.png'.format(img_path, prefix, self.itr), nrow=10)

        sample_and_save('normal')
        if self.config.use_ema:
            G_state = copy.deepcopy(self.netG.state_dict())
            self.netG.load_state_dict(self.ema.target_dict)
            sample_and_save('ema')
            self.netG.load_state_dict(G_state)

    def compute_inception_fid(self):
        self.netG.eval()
        if self.config.use_ema:
            G_state = copy.deepcopy(self.netG.state_dict())
            self.netG.load_state_dict(self.ema.target_dict)
        
        # bs = self.config.batchSize
        bs = 32
        samples = []
        labels_gen = []
        num_batches = int(self.config.num_inception_imgs / bs)
        for batch in range(num_batches):
            with torch.no_grad():
                z = utils.sample_normal(bs, self.config.nz, device=self.device)
                if self.config.conditional:
                    y = utils.sample_cats(bs, self.num_classes, device=self.device)
                    labels_gen.append(y.cpu().numpy())
                else:
                    y = None

                gen = self.netG(z, y)
                gen = gen * 0.5 + 0.5
                gen = gen * 255.0
                gen = gen.cpu().numpy().astype(np.uint8)
                gen = np.transpose(gen, (0, 2, 3, 1))
                samples.extend(gen)

        if self.config.conditional:
            labels_gen = np.hstack(labels_gen)
            samples = (samples, labels_gen)
            IS_mean, IS_std, fid, intra_fid = self.inception_evaluator.compute_metrics(samples)
            self.log('IS: {} +/- {}'.format(IS_mean, IS_std))
            self.log('FID: {}'.format(fid))
            self.log('Intra FID: {}'.format(intra_fid))

            # Choosing the min FID model
            if self.best_intra_fid > intra_fid:
                self.best_is = IS_mean
                self.best_is_std = IS_std
                self.best_fid = fid
                self.best_intra_fid = intra_fid
                self.save_state('model_best.pth')
        else:
            IS_mean, IS_std, fid = self.inception_evaluator.compute_metrics(samples)
            self.log('IS: {} +/- {}'.format(IS_mean, IS_std))
            self.log('FID: {}'.format(fid))

            # Choosing the min FID model
            if self.best_fid > fid:
                self.best_is = IS_mean
                self.best_is_std = IS_std
                self.best_fid = fid
                self.save_state('model_best.pth')

        self.log('Best IS: {} +/- {}'.format(self.best_is, self.best_is_std))
        self.log('Best FID: {}'.format(self.best_fid))
        if self.config.conditional:
            self.log('Best intra FID: {}'.format(self.best_intra_fid))

        if self.config.use_ema:
            self.netG.load_state_dict(G_state)

    def benchmark_inception_fid(self):
        samples = []

        for dat in self.testloader:
            imgs, labels = dat
            imgs = imgs * 0.5 + 0.5
            imgs = imgs * 255.0
            imgs = imgs.numpy().astype(np.uint8)
            imgs = np.transpose(imgs, (0, 2, 3, 1))
            samples.extend(imgs)

        IS_mean, IS_std, fid = self.inception_evaluator.compute_metrics(samples)
        self.log('Real data IS: {} +/- {}'.format(IS_mean, IS_std))
        self.log('Real data FID: {}'.format(fid))

    def train(self):
        # self.benchmark_inception_fid()
        while self.epoch < self.config.nepochs:
            for i, data in enumerate(self.dataloader, 0):

                self.netD.train()
                self.netG.train()

                # Forming data and label tensors
                real_data, real_labels, data_indices = data
                real_data = real_data.to(self.device)
                real_labels = real_labels.to(self.device)

                if not self.config.conditional:
                    real_labels = None

                # Weight updates
                if self.weight_update_flag:
                    if (self.itr % self.config.weight_update_iters == 0) and self.itr > 300:
                        self.weight_updates(real_data, real_labels, (self.itr % 3000 == 0))

                # Updates
                disc_loss, gen_loss = self.gan_updates(real_data, real_labels, data_indices)
                if self.config.use_ema:
                    self.ema.update(self.itr)

                self.itr += 1

                if self.itr == 1:
                    img_path = osp.join(self.config.logdir, 'samples')
                    Path(img_path).mkdir(parents=True, exist_ok=True)
                    vutils.save_image(real_data * 0.5 + 0.5,
                                      '{}/real.png'.format(img_path))

                if self.itr % self.config.log_every == 0:
                    lrG = self.optimizerG.param_groups[0]['lr']
                    lrD = self.optimizerD.param_groups[0]['lr']
                    self.log(
                        '[{}/{}] Iteration: {}, Disc loss: {}, Gen loss: '
                        '{}, lrG: {}, lrD: {}'.format(self.epoch, self.config.nepochs, self.itr,
                                    disc_loss, gen_loss, lrG, lrD))

                if self.itr % self.config.test_every == 0:
                    self.save_samples()
                    self.save_state()

                if self.itr % self.config.test_inception_every == 0:
                    self.compute_inception_fid()

            if self.weight_update_flag and self.weight_update_type == 'discrete':
                # Discriminator momentum
                if self.epoch == 0:
                    self.disc_vector = self.disc_vector_cur
                else:
                    self.disc_vector = self.disc_vector_cur * (1 - self.config.disc_momentum) + \
                                       self.config.disc_momentum * self.disc_vector
                self.disc_vector_cur.fill_(0)

            self.epoch += 1
