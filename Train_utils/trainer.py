import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch.optim import Adam
from pathlib import Path
from tqdm.auto import tqdm
from ema_pytorch import EMA
from torch.nn.utils import clip_grad_norm_
from Utils.em_utils import expectation_maximization
from Train_utils.lr_schedule import ReduceLROnPlateauWithWarmup


def cycle(dl):
    while True:
        for data in dl:
            yield data


class Trainer(object):
    def __init__(
        self,
        tmc_model,
        # ae_model,
        model, 
        data_loader, 
        results_folder='./Checkpoints', 
        train_lr=1e-5, 
        warmup_lr=1e-4, 
        save_cycle=10000,
        train_num_steps=100000, 
        adam_betas=(0.9, 0.96), 
        gradient_accumulate_every=2, 
        ema_update_every=10,
        ema_decay=0.995,
        patience=1000, 
        min_lr=1e-6, 
        threshold=0., 
        warmup=2000,
        factor=0.5,
        ratio=0.1
    ):
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        # self.ae_model = ae_model.to(self.device)
        self.tmc_model = tmc_model.to(self.device)
        self.A = self.tmc_model.rm.to(self.device)
        self.weight = 0.1 * ratio / 2

        self.pre_epoch = train_num_steps
        self.train_num_steps = train_num_steps

        self.gradient_accumulate_every = gradient_accumulate_every
        self.save_cycle = int(save_cycle)
        self.dataloader = data_loader
        self.dl = cycle(data_loader)
        self.step = 0
        self.milestone = 0

        self.results_folder = Path(results_folder)
        os.makedirs(self.results_folder, exist_ok=True)

        self.opt = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=train_lr, betas=adam_betas)
        self.sch = ReduceLROnPlateauWithWarmup(optimizer=self.opt, factor=factor, patience=patience, min_lr=min_lr, threshold=threshold,
                                               threshold_mode='rel', warmup_lr=warmup_lr, warmup=warmup, verbose=False)
        
        # self.ae_opt = Adam(filter(lambda p: p.requires_grad, self.ae_model.parameters()), lr=1e-3, betas=adam_betas)
        self.criteon = nn.L1Loss().to(self.device)
        self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_every).to(self.device)

        self.tmc_opt = Adam(filter(lambda p: p.requires_grad, self.tmc_model.parameters()), lr=1e-3, betas=adam_betas)

    def save(self, milestone):
        if milestone < 10:
            return
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            # 'ae_model': self.ae_model.state_dict(),
            'tmc_model': self.tmc_model.state_dict(),
            'ema': self.ema.state_dict(),
            'opt': self.opt.state_dict(),
        }
        torch.save(data, str(self.results_folder / f'checkpoint-{milestone}.pt'))

    def load(self, milestone):
        device = self.device
        self.milestone = milestone
        data = torch.load(str(self.results_folder / f'checkpoint-{milestone}.pt'), map_location=device)
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])
        self.model.load_state_dict(data['model'])
        # self.ae_model.load_state_dict(data['ae_model'])
        self.tmc_model.load_state_dict(data['tmc_model'])

    def train(self):
        device = self.device
        step = 0
        
        print("Start TMC Network Training.")

        with tqdm(initial=step, total=self.pre_epoch) as pbar:

            while step < self.pre_epoch:

                x, _, m, _ = next(self.dl)
                x, m = x.to(device), m.to(device)
                x_hat = self.tmc_model(x * m)
                # x_hat = self.tmc_model(x)
                loss = self.criteon(x_hat * m, x * m)
                loss.backward()

                pbar.set_description(f'loss: {loss.item():.6f}')
                self.tmc_opt.step()
                self.tmc_opt.zero_grad()

                step += 1
                pbar.update(1)

        self.tmc_model.requires_grad_(False)
        self.tmc_model.eval()
        print("Finish TMC Network Training. Now Start Optimizing.")

        restoration = np.empty([0, self.dataloader.dataset.window, self.dataloader.dataset.dim_2])
        for idx, (x, y, m1, m2) in enumerate(self.dataloader):
            x, y, m1, m2 = x.to(self.device), y.to(self.device), m1.to(self.device), m2.to(self.device)
            x_hat = self.tmc_model.preprocessing(x * m1, x * m1, m1)
            restoration = np.row_stack([restoration, x_hat.detach().cpu().numpy()])
        restoration = torch.from_numpy(restoration).float()

        print("Finish Optimizing. Now Start Joint Training.")

        self.dataloader.dataset.update(restoration)
        self.dl_update = cycle(self.dataloader)
        step = 0

        with tqdm(initial=step, total=self.train_num_steps) as pbar:
            while step < self.train_num_steps:
                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    x, y, m1, m2 = next(self.dl_update)
                    x, y, m1, m2 = x.to(device), y.to(device), m1.to(device), m2.to(device)
                    x_hat = self.model(x)
                    loss = self.criteon(x_hat * m1, x * m1)
                    loss = loss / self.gradient_accumulate_every
                    loss.backward()
                    total_loss += loss.item()

                pbar.set_description(f'loss: {total_loss:.6f}')

                clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.sch.step(total_loss)
                self.opt.zero_grad()
                step += 1
                self.step += 1
                self.ema.update()

                with torch.no_grad():
                    if self.step != 0 and self.step % self.save_cycle == 0:
                        self.milestone += 1
                        self.save(self.milestone)

                pbar.update(1)

        print('training complete')

    def sample(self, num, size_every, size):
        samples = np.empty([0, size[0], size[1]])
        num_cycle = int(num // size_every) + 1

        for _ in range(num_cycle):
            sample = self.ema.ema_model.sample(batch_size=size_every)
            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            torch.cuda.empty_cache()

        return samples
    
    def estimate(self, dataloader, A, A_pinv, learning_rate=5e-2):
        test_loss = []
        model_kwargs = {}
        model_kwargs['learning_rate'] = learning_rate
        A, A_pinv = A.to(self.device), A_pinv.to(self.device)
        estimations = np.empty([0, dataloader.dataset.dim_2])
        reals = np.empty([0, dataloader.dataset.dim_2])
        self.ema.ema_model.eval()
        self.tmc_model.eval()
        for idx, (x, y, m) in enumerate(dataloader):
            x, y, m = x.to(self.device), y.to(self.device), m.to(self.device)
            x_hat = self.ema.ema_model.traffic_matrix_estimate(A, y, m, model_kwargs)
            x_hat = expectation_maximization(x_hat, y, A, 5)
            
            estimations = np.row_stack([estimations, x_hat.reshape(-1, x_hat.shape[-1]).detach().cpu().numpy()])
            reals = np.row_stack([reals, x.reshape(-1, x.shape[-1]).detach().cpu().numpy()])
            torch.cuda.empty_cache()

            test_loss_x = self.criteon(x_hat, x)
            test_loss.append(test_loss_x.item())

        test_loss = np.average(test_loss)
        print('Testing Mean Error:', test_loss.item())
        return estimations, reals
    
    def complete(self, dataloader, learning_rate=5e-2):
        test_loss = []
        model_kwargs = {}
        model_kwargs['learning_rate'] = learning_rate
        completions = np.empty([0, dataloader.dataset.dim])
        reals = np.empty([0, dataloader.dataset.dim])
        masks = np.empty([0, dataloader.dataset.dim])
        self.ema.ema_model.eval()
        for idx, (x, m1, m2) in enumerate(dataloader):
            x, m1, m2 = x.to(self.device), m1.to(self.device), m2.to(self.device)
            m = m1 * (1 - m2)
            x_hat = self.ema.ema_model.traffic_matrix_complete(x * m2, m2, model_kwargs)
            completions = np.row_stack([completions, x_hat.reshape(-1, x_hat.shape[-1]).detach().cpu().numpy()])
            reals = np.row_stack([reals, x.reshape(-1, x.shape[-1]).detach().cpu().numpy()])
            masks = np.row_stack([masks, m.reshape(-1, m.shape[-1]).detach().cpu().numpy()])
            torch.cuda.empty_cache()

            test_loss_x = self.criteon(x_hat * m, x * m)
            test_loss.append(test_loss_x.item())

        test_loss = np.average(test_loss)
        print('Testing Mean Error:', test_loss.item())
        return completions, masks, reals
    
    def combine(self, dataloader, A, A_pinv, learning_rate=5e-2, coef=0.):
        test_loss = []
        model_kwargs = {}
        model_kwargs['coef'] = coef
        model_kwargs['learning_rate'] = learning_rate
        completions = np.empty([0, dataloader.dataset.dim_2])
        reals = np.empty([0, dataloader.dataset.dim_2])
        masks = np.empty([0, dataloader.dataset.dim_2])
        A, A_pinv = A.to(self.device), A_pinv.to(self.device)
        self.ema.ema_model.eval()
        for idx, (x, y, my, m1, m2) in enumerate(dataloader):
            x, m1, m2 = x.to(self.device), m1.to(self.device), m2.to(self.device)
            y, my = y.to(self.device), my.to(self.device)
            m = m1 * (1 - m2)
            x_hat = self.ema.ema_model.traffic_matrix_combine(x * m2, m2, A, y * my, my, model_kwargs)
            args = (my > 0)[0, 0, :]
            x_hat = expectation_maximization(x_hat, y[:, :, args], A[:, args], 5, m2)
            completions = np.row_stack([completions, x_hat.reshape(-1, x_hat.shape[-1]).detach().cpu().numpy()])
            reals = np.row_stack([reals, x.reshape(-1, x.shape[-1]).detach().cpu().numpy()])
            masks = np.row_stack([masks, m.reshape(-1, m.shape[-1]).detach().cpu().numpy()])
            torch.cuda.empty_cache()

            test_loss_x = self.criteon(x_hat * m, x * m)
            test_loss.append(test_loss_x.item())

        test_loss = np.average(test_loss)
        print('Testing Mean Error:', test_loss.item())
        return completions, masks, reals


