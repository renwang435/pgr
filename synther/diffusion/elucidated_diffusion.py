import math
import pathlib
from multiprocessing import cpu_count
from typing import Optional, Sequence, Tuple

import gin
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from einops import reduce
from ema_pytorch import EMA
from redq.algos.core import ReplayBuffer
from synther.diffusion.norm import BaseNormalizer
from synther.online.utils import make_inputs_from_replay_buffer
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


# helpers
def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


# tensor helpers
def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


# main class
@gin.configurable
class ElucidatedDiffusion(nn.Module):
    def __init__(
            self,
            net,
            normalizer: BaseNormalizer,
            cond_normalizer: BaseNormalizer,
            event_shape: Sequence[int],  # shape of the input and output
            num_sample_steps: int = 32,  # number of sampling steps
            sigma_min: float = 0.002,  # min noise level
            sigma_max: float = 80,  # max noise level
            sigma_data: float = 1.0,  # standard deviation of data distribution
            rho: float = 7,  # controls the sampling schedule
            P_mean: float = -1.2,  # mean of log-normal distribution from which noise is drawn for training
            P_std: float = 1.2,  # standard deviation of log-normal distribution from which noise is drawn for training
            S_churn: float = 80,  # parameters for stochastic sampling - depends on dataset, Table 5 in paper
            S_tmin: float = 0.05,
            S_tmax: float = 50,
            S_noise: float = 1.003,
    ):
        super().__init__()
        assert net.random_or_learned_sinusoidal_cond
        self.net = net
        self.normalizer = normalizer
        self.cond_normalizer = cond_normalizer

        # input dimensions
        self.event_shape = event_shape

        # parameters
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.P_mean = P_mean
        self.P_std = P_std
        self.num_sample_steps = num_sample_steps  # otherwise known as N in the paper
        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise

    @property
    def device(self):
        return next(self.net.parameters()).device

    # derived preconditioning params - Table 1
    def c_skip(self, sigma):
        return (self.sigma_data ** 2) / (sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma):
        return sigma * self.sigma_data * (self.sigma_data ** 2 + sigma ** 2) ** -0.5

    def c_in(self, sigma):
        return 1 * (sigma ** 2 + self.sigma_data ** 2) ** -0.5

    def c_noise(self, sigma):
        return log(sigma) * 0.25

    # preconditioned network output, equation (7) in the paper
    def preconditioned_network_forward(self, noised_inputs, sigma, clamp=False, cond=None):
        batch, device = noised_inputs.shape[0], noised_inputs.device

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device=device)

        padded_sigma = sigma.view(batch, *([1] * len(self.event_shape)))

        net_out = self.net(
            self.c_in(padded_sigma) * noised_inputs,
            self.c_noise(sigma),
            cond=cond,
        )

        out = self.c_skip(padded_sigma) * noised_inputs + self.c_out(padded_sigma) * net_out

        if clamp:
            out = out.clamp(-1., 1.)

        return out

    # sample schedule, equation (5) in the paper
    def sample_schedule(self, num_sample_steps=None):
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        N = num_sample_steps
        inv_rho = 1 / self.rho

        steps = torch.arange(num_sample_steps, device=self.device, dtype=torch.float32)
        sigmas = (self.sigma_max ** inv_rho + steps / (N - 1) * (
                self.sigma_min ** inv_rho - self.sigma_max ** inv_rho)) ** self.rho

        sigmas = F.pad(sigmas, (0, 1), value=0.)  # last step is sigma value of 0.
        return sigmas

    @torch.no_grad()
    def sample(
            self,
            batch_size: int = 16,
            num_sample_steps: Optional[int] = None,
            clamp: bool = True,
            cond=None,
            cfg_scale: float = 1.0,
            disable_tqdm: bool = False,
    ):

        num_sample_steps = default(num_sample_steps, self.num_sample_steps)
        shape = (batch_size, *self.event_shape)

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma
        sigmas = self.sample_schedule(num_sample_steps)
        gammas = torch.where(
            (sigmas >= self.S_tmin) & (sigmas <= self.S_tmax),
            min(self.S_churn / num_sample_steps, math.sqrt(2) - 1),
            0.
        )

        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))

        # inputs are noise at the beginning
        init_sigma = sigmas[0]
        inputs = init_sigma * torch.randn(shape, device=self.device)
        if cond is not None:
            cond = torch.from_numpy(cond).float().to(self.device)
            cond = self.cond_normalizer.normalize(cond)

        # gradually denoise
        for sigma, sigma_next, gamma in tqdm(sigmas_and_gammas, desc='sampling time step', mininterval=1,
                                             disable=disable_tqdm):
            sigma, sigma_next, gamma = map(lambda t: t.item(), (sigma, sigma_next, gamma))

            eps = self.S_noise * torch.randn(shape, device=self.device)  # stochastic sampling

            sigma_hat = sigma + gamma * sigma
            inputs_hat = inputs + math.sqrt(sigma_hat ** 2 - sigma ** 2) * eps

            # denoised_over_sigma = self.score_fn(inputs_hat, sigma_hat, clamp=clamp, cond=cond)
            # w/ cond
            cond_denoised_over_sigma = self.score_fn(inputs_hat, sigma_hat, clamp=clamp, cond=cond)
            # w/o cond
            uncond_denoised_over_sigma = self.score_fn(inputs_hat, sigma_hat, clamp=clamp, cond=None)
            # do cfg
            denoised_over_sigma = uncond_denoised_over_sigma + cfg_scale * (cond_denoised_over_sigma - uncond_denoised_over_sigma)

            inputs_next = inputs_hat + (sigma_next - sigma_hat) * denoised_over_sigma

            # second order correction, if not the last timestep
            if sigma_next != 0:
                if cfg_scale == 0.0:
                    denoised_prime_over_sigma = self.score_fn(inputs_next, sigma_next, clamp=clamp, cond=None)
                else:
                    denoised_prime_over_sigma = self.score_fn(inputs_next, sigma_next, clamp=clamp, cond=cond)
                inputs_next = inputs_hat + 0.5 * (sigma_next - sigma_hat) * (
                        denoised_over_sigma + denoised_prime_over_sigma)

            inputs = inputs_next

        if clamp:
            inputs = inputs.clamp(-1., 1.)
        return self.normalizer.unnormalize(inputs)

    # This is known as 'denoised_over_sigma' in the lucidrains repo.
    def score_fn(
            self,
            x,
            sigma,
            clamp: bool = False,
            cond=None,
    ):
        denoised = self.preconditioned_network_forward(x, sigma, clamp=clamp, cond=cond)
        denoised_over_sigma = (x - denoised) / sigma

        return denoised_over_sigma

    # training
    def loss_weight(self, sigma):
        return (sigma ** 2 + self.sigma_data ** 2) * (sigma * self.sigma_data) ** -2

    def noise_distribution(self, batch_size):
        return (self.P_mean + self.P_std * torch.randn((batch_size,), device=self.device)).exp()

    def forward(self, inputs, cond=None):
        inputs = self.normalizer.normalize(inputs)
        if cond is not None:
            cond = self.cond_normalizer.normalize(cond)

        batch_size, *event_shape = inputs.shape
        assert event_shape == self.event_shape, f'mismatch of event shape, ' \
                                                f'expected {self.event_shape}, got {event_shape}'

        sigmas = self.noise_distribution(batch_size)
        padded_sigmas = sigmas.view(batch_size, *([1] * len(self.event_shape)))

        noise = torch.randn_like(inputs)
        noised_inputs = inputs + padded_sigmas * noise  # alphas are 1. in the paper

        denoised = self.preconditioned_network_forward(noised_inputs, sigmas, cond=cond)
        losses = F.mse_loss(denoised, inputs, reduction='none')
        losses = reduce(losses, 'b ... -> b', 'mean')
        losses = losses * self.loss_weight(sigmas)
        return losses.mean()


@gin.configurable
class Trainer(object):
    def __init__(
            self,
            diffusion_model,
            dataset: Optional[torch.utils.data.Dataset] = None,
            train_batch_size: int = 16,
            small_batch_size: int = 16,
            gradient_accumulate_every: int = 1,
            train_lr: float = 1e-4,
            lr_scheduler: Optional[str] = None,
            train_num_steps: int = 100000,
            ema_update_every: int = 10,
            ema_decay: float = 0.995,
            adam_betas: Tuple[float, float] = (0.9, 0.99),
            save_and_sample_every: int = 10000,
            weight_decay: float = 0.,
            results_folder: str = './results',
            amp: bool = False,
            fp16: bool = False,
            split_batches: bool = True,
    ):
        super().__init__()
        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no'
        )
        self.accelerator.native_amp = amp
        self.model = diffusion_model

        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.num_params = num_params
        print(f'Number of trainable parameters: {num_params}.')

        self.save_and_sample_every = save_and_sample_every
        self.train_num_steps = train_num_steps
        self.gradient_accumulate_every = gradient_accumulate_every

        if dataset is not None:
            # If dataset size is less than 800K use the small batch size
            if len(dataset) < int(8e5):
                self.batch_size = small_batch_size
            else:
                self.batch_size = train_batch_size
            print(f'Using batch size: {self.batch_size}')
            # dataset and dataloader
            dl = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=cpu_count())
            dl = self.accelerator.prepare(dl)
            self.dl = cycle(dl)
        else:
            # No dataloader, train batch by batch
            self.batch_size = train_batch_size
            self.dl = None

        # optimizer, make sure that the bias and layer-norm weights are not decayed
        no_decay = ['bias', 'LayerNorm.weight', 'norm.weight', '.g']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay,
            },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        self.opt = torch.optim.AdamW(optimizer_grouped_parameters, lr=train_lr, betas=adam_betas)

        # for logging results in a folder periodically
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
            self.results_folder = pathlib.Path(results_folder)
            self.results_folder.mkdir(exist_ok=True)

        # step counter state
        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        if lr_scheduler == 'linear':
            print('using linear learning rate scheduler')
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.opt,
                lambda step: max(0, 1 - step / train_num_steps)
            )
        elif lr_scheduler == 'cosine':
            print('using cosine learning rate scheduler')
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.opt,
                train_num_steps
            )
        else:
            self.lr_scheduler = None

        self.model.normalizer.to(self.accelerator.device)
        self.ema.ema_model.normalizer.to(self.accelerator.device)

        if self.model.cond_normalizer is not None:
            self.model.cond_normalizer.to(self.accelerator.device)
            self.ema.ema_model.cond_normalizer.to(self.accelerator.device)

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone: int):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    # Train for the full number of steps.
    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = (next(self.dl)[0]).to(device)

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.save(self.step)

                pbar.update(1)

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

        accelerator.print('training complete')

    # Allow user to pass in external data.
    # NOTE: THIS ASSUMES WE ARE TRAINING ON A SINGLE GPU
    def train_on_batch(
            self,
            data: torch.Tensor,
            **kwargs,
    ):
        accelerator = self.accelerator
        device = accelerator.device
        data = data.to(device)
        if 'cond' in kwargs:
            kwargs['cond'] = kwargs['cond'].to(device)

        total_loss = 0.
        with self.accelerator.autocast():
            loss = self.model(data, **kwargs)
            total_loss += loss.item()
        self.accelerator.backward(loss)
        accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

        self.opt.step()
        self.opt.zero_grad()

        self.step += 1
        self.ema.to(device)
        self.ema.update()

        if self.step != 0 and self.step % self.save_and_sample_every == 0:
            self.save(self.step)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return total_loss


@gin.configurable
class REDQCondTrainer(Trainer):
    def __init__(
            self,
            diffusion_model,
            train_batch_size: int = 16,
            gradient_accumulate_every: int = 1,
            train_lr: float = 1e-4,
            lr_scheduler: Optional[str] = None,
            train_num_steps: int = 100000,
            ema_update_every: int = 10,
            ema_decay: float = 0.995,
            adam_betas: Tuple[float, float] = (0.9, 0.99),
            save_and_sample_every: int = 10000,
            weight_decay: float = 0.,
            results_folder: str = './results',
            amp: bool = False,
            fp16: bool = False,
            split_batches: bool = True,
            model_terminals: bool = False,
    ):
        super().__init__(
            diffusion_model,
            dataset=None,
            train_batch_size=train_batch_size,
            gradient_accumulate_every=gradient_accumulate_every,
            train_lr=train_lr,
            lr_scheduler=lr_scheduler,
            train_num_steps=train_num_steps,
            ema_update_every=ema_update_every,
            ema_decay=ema_decay,
            adam_betas=adam_betas,
            save_and_sample_every=save_and_sample_every,
            weight_decay=weight_decay,
            results_folder=results_folder,
            amp=amp,
            fp16=fp16,
            split_batches=split_batches,
        )

        self.model_terminals = model_terminals
    
    def save_final(self, cond_distri, epoch, milestone):
        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }

        torch.save(data, str(self.results_folder / f'model-ep{epoch}-step{milestone}.pt'))

        # Save the conditional distribution
        # only need irews_buf and top_frac_indices
        np.save(str(self.results_folder / f'irews_buf-ep{epoch}-step{milestone}.npy'), cond_distri.irews_buf)
        np.save(str(self.results_folder / f'top_frac_indices-ep{epoch}-step{milestone}.npy'), cond_distri.top_frac_indices)

    def load_final(self, epoch, milestone: int):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-ep{epoch}-step{milestone}.pt'), map_location=device)

        irews_buf = np.load(str(self.results_folder / f'irews_buf-ep{epoch}-step{milestone}.npy'))
        top_frac_indices = np.load(str(self.results_folder / f'top_frac_indices-ep{epoch}-step{milestone}.npy'))

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])
        
        return irews_buf, top_frac_indices

    def train_from_redq_buffer(self, buffer: ReplayBuffer, cond_net, top_frac, 
                               curr_epoch: int,
                               num_steps: Optional[int] = None):
        cond_net.eval()
        cond_distri = CondDistri(cond_net, self.batch_size, buffer, top_frac)
        self.update_cond_normalizer(cond_distri, device=self.accelerator.device)

        num_steps = num_steps or self.train_num_steps
        for j in range(num_steps):
            b = cond_distri.sample_batch(self.batch_size)
            obs = b['obs1']
            next_obs = b['obs2']
            actions = b['acts']
            rewards = b['rews'][:, None]
            done = b['done'][:, None]
            cond_signal = b['irews'][:, None]

            data = [obs, actions, rewards, next_obs]
            if self.model_terminals:
                data.append(done)
            data = np.concatenate(data, axis=1)
            data = torch.from_numpy(data).float()
            cond_signal = torch.from_numpy(cond_signal).float()
            loss = self.train_on_batch(data, cond=cond_signal)
            if j % 1000 == 0:
                print(f'[{j}/{num_steps}] loss: {loss:.4f}')
        
        self.save_final(cond_distri, curr_epoch, num_steps)
        
        return cond_distri
    
    def update_cond_normalizer(self, cond_distri, device=None):
        data = cond_distri.irews_buf[:, None]
        data = torch.from_numpy(data).float()
        self.model.cond_normalizer.reset(data)
        self.ema.ema_model.cond_normalizer.reset(data)
        if device:
            self.model.cond_normalizer.to(device)
            self.ema.ema_model.cond_normalizer.to(device)

    def update_normalizer(self, buffer: ReplayBuffer, device=None):
        data = make_inputs_from_replay_buffer(buffer, self.model_terminals)
        data = torch.from_numpy(data).float()
        self.model.normalizer.reset(data)
        self.ema.ema_model.normalizer.reset(data)
        if device:
            self.model.normalizer.to(device)
            self.ema.ema_model.normalizer.to(device)



class CondDistri(object):
    def __init__(self, cond_net, train_batch_size, buffer, top_frac):
        self.top_frac = top_frac
        self.buffer = buffer

        # Iterate over buffer and generate the conditional generation signal
        self.irews_buf = np.zeros_like(buffer.rews_buf)
        # Use some large batch size
        for i in range(0, buffer.size, train_batch_size):
            idxs = np.arange(i, min(i + train_batch_size, buffer.size))
            obs = self.buffer.obs1_buf[idxs]
            next_obs = self.buffer.obs2_buf[idxs]
            actions = self.buffer.acts_buf[idxs]
            rewards = self.buffer.rews_buf[idxs][:, None]
            done = self.buffer.done_buf[idxs][:, None]
            with torch.no_grad():
                self.irews_buf[idxs] = cond_net.compute_reward(obs, next_obs, actions, rewards, done).squeeze().cpu().numpy()
        
        self.top_frac_indices = np.argsort(self.irews_buf, axis=0)[-int(top_frac * buffer.size):]

    def sample_batch(self, batch_size=32, idxs=None):
        """
        :param batch_size: size of minibatch
        :param idxs: specify indexes if you want specific data points
        :return: mini-batch data as a dictionary
        """
        if idxs is None:
            idxs = np.random.randint(0, self.buffer.size, size=batch_size)
        return dict(obs1=self.buffer.obs1_buf[idxs],
                    obs2=self.buffer.obs2_buf[idxs],
                    acts=self.buffer.acts_buf[idxs],
                    rews=self.buffer.rews_buf[idxs],
                    done=self.buffer.done_buf[idxs],
                    irews=self.irews_buf[idxs],
                    idxs=idxs)

    def sample_uncond(self, batch_size):
        # Sample batch_size randomly from self.irews
        return self.irews_buf[np.random.choice(self.irews_buf.shape[0], batch_size, replace=True), None]
    
    def sample_cond(self, batch_size):
        # Sample from top_frac highest values of self.irews
        best_indices = np.random.choice(self.top_frac_indices, batch_size, replace=True)
        return self.irews_buf[best_indices, None]
