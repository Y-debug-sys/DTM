import math
import torch
import torch.nn.functional as F

from torch import nn
from random import random
from tqdm.auto import tqdm
from functools import partial
from collections import namedtuple


ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.00001
    beta_end = scale * 0.0005
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        seq_length,
        *,
        timesteps = 1000,
        sampling_timesteps = None,
        loss_type = 'l1',
        objective = 'pred_noise',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        # p2_loss_weight_gamma = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        # p2_loss_weight_k = 1,
        ddim_sampling_eta = 0.
    ):
        super().__init__()
        self.model = model
        self.self_condition = False
        self.feature_size = self.model.feature_size
        self.seq_length = seq_length

        self.objective = objective
        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0) # Returns the cumulative product of elements of input in the dimension dim
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond=None, clip_x_start=True):
        model_output = self.model(x, t, x_self_cond)
        maybe_clip = partial(torch.clamp, min = 0., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(0., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, x_self_cond = None, clip_denoised = True):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = \
            self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)

        # img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def ddim_sample(self, shape, clip_denoised = True):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        return img

    @torch.no_grad()
    def sample(self, batch_size=16, model=None):
        seq_length, feature_size = self.seq_length, self.feature_size
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        traffic = sample_fn((batch_size, seq_length, feature_size))
        if model is not None:
            return model.recover(traffic)
        return traffic

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    
    def traffic_matrix_complete(self, target, masks, model_kwargs):
        batch_size = target.shape[0]
        feature_size, channels = self.feature_size, self.seq_length
        tmc_fn = self.TMC if not self.is_ddim_sampling else self.fast_TMC
        traffic = tmc_fn((batch_size, channels, feature_size), target, masks, model_kwargs=model_kwargs)
        return traffic

    def traffic_matrix_estimate(self, A, link, masks, model_kwargs=None):
        batch_size = link.shape[0]
        feature_size, channels = self.feature_size, self.seq_length
        tme_fn = self.TME if not self.is_ddim_sampling else self.fast_TME
        traffic = tme_fn((batch_size, channels, feature_size), A, link, masks, model_kwargs=model_kwargs)
        return traffic
    
    def traffic_matrix_combine(self, target, target_masks, A, link, link_masks, model_kwargs):
        batch_size = target.shape[0]
        feature_size, channels = self.feature_size, self.seq_length
        tm_fn = self.TM if not self.is_ddim_sampling else self.fast_TM
        traffic = tm_fn((batch_size, channels, feature_size), target, target_masks, A,
                        link, link_masks, model_kwargs=model_kwargs)
        return traffic
    
    def TM(self, shape, target, partial_mask_1, A, link, partial_mask_2, model_kwargs, clip_denoised=True):
        device = self.betas.device
        x = torch.randn(shape, device=device)
        x_start = None
        
        m1 = (partial_mask_1>0).reshape(partial_mask_1.shape).to(device)
        m2 = (partial_mask_2>0).reshape(partial_mask_2.shape).to(device)

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='estimation sample time step', total=self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            batched_times = torch.full((x.shape[0],), t, device=device, dtype=torch.long)

            preds = self.model_predictions(x, batched_times, self_cond, clip_x_start=clip_denoised)
            x_start = preds.pred_x_start

            model_mean, _, model_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=batched_times)
            noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
            sigma = (0.5 * model_log_variance).exp()
            x = model_mean + sigma * noise

            x = self.guidance_both(A=A, tgt_embs_link=link, tgt_embs_flow=target, partial_mask_flow=m1, 
                                         partial_mask_link=m2, sample=x, t=batched_times, **model_kwargs)
            
            target_t = self.q_sample(target, t=batched_times)
            x[m1] = target_t[m1]

        traffic = torch.clamp_min(x, 0.)
        traffic[m1] = target[m1]
        return traffic
    
    def fast_TM(self, shape, target, partial_mask_1, A, link, partial_mask_2, model_kwargs, clip_denoised=True):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = torch.randn(shape, device = device)
        m1 = (partial_mask_1>0).reshape(partial_mask_1.shape).to(device)
        m2 = (partial_mask_2>0).reshape(partial_mask_2.shape).to(device)
        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(x, time_cond, self_cond, clip_x_start=clip_denoised)

            if time_next < 0:
                x = x_start
                continue

            x_start_hat = torch.clamp_min(x_start, 0.)
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x)
            pred_mean = x_start_hat * alpha_next.sqrt() + c * pred_noise
            x = pred_mean + sigma * noise
            x = self.guidance_both(A=A, tgt_embs_link=link, tgt_embs_flow=target, partial_mask_flow=m1, 
                                   partial_mask_link=m2, sample=x, t=time_cond, **model_kwargs)
            
            target_t = self.q_sample(target, t=time_cond)
            x[m1] = target_t[m1]

        traffic = torch.clamp_min(x, 0.)
        traffic[m1] = target[m1]
        return torch.clamp_min(x, 0.)
    
    def TMC(self, shape, target, partial_mask, model_kwargs):
        device = self.betas.device
        partial_mask = (partial_mask>0).reshape(partial_mask.shape).to(device)
        x = torch.randn(shape, device=device)
        for t in tqdm(reversed(range(0, self.num_timesteps)),
                      desc='sampling loop time step', total=self.num_timesteps):
            x = self.p_tmc(x=x, t=t, target=target, partial_mask=partial_mask, model_kwargs=model_kwargs)
        traffic = torch.clamp_min(x, 0.)
        traffic[partial_mask] = target[partial_mask]
        return traffic

    def p_tmc(self, x, target, t, partial_mask, model_kwargs, clip_denoised=True):
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        preds = self.model_predictions(x, batched_times, clip_x_start=clip_denoised)
        x_start = preds.pred_x_start

        model_mean, _, model_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=batched_times)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        sigma = (0.5 * model_log_variance).exp()
        x = model_mean + sigma * noise

        x = self.guidance_tmc(sample=x, t=batched_times, tgt_embs=target, partial_mask=partial_mask, **model_kwargs)
        target_t = self.q_sample(target, t=batched_times)
        x[partial_mask] = target_t[partial_mask]
        return x

    def fast_TMC(self, shape, target, partial_mask, model_kwargs, clip_denoised=True):
        batch, device, total_timesteps, eta = shape[0], self.betas.device, self.num_timesteps, self.eta
        partial_mask = (partial_mask>0).reshape(partial_mask.shape).to(device)

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=self.sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        x = torch.randn(shape, device=device)

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(x, time_cond, clip_x_start=clip_denoised)

            if time_next < 0:
                x = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            pred_mean = x_start * alpha_next.sqrt() + c * pred_noise
            noise = torch.randn_like(x)

            x = pred_mean + sigma * noise
            x = self.guidance_tmc(sample=x, t=time_cond, tgt_embs=target, partial_mask=partial_mask, **model_kwargs)

            target_t = self.q_sample(target, t=time_cond)
            x[partial_mask] = target_t[partial_mask]

        traffic = torch.clamp_min(x, 0.)
        traffic[partial_mask] = target[partial_mask]
        return traffic
    
    def TME(self, shape, A, link, masks, clip_denoised=True, model_kwargs=None):
        device = self.betas.device
        x = torch.randn(shape, device=device)
        x_start = None
        
        m = (masks>0).reshape(masks.shape).to(device)
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='estimation sample time step', total=self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            batched_times = torch.full((x.shape[0],), t, device=device, dtype=torch.long)

            preds = self.model_predictions(x, batched_times, self_cond, clip_x_start=clip_denoised)
            x_start = preds.pred_x_start

            model_mean, _, model_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=batched_times)
            noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
            sigma = (0.5 * model_log_variance).exp()
            x = model_mean + sigma * noise
            x = self.guidance_nt(sample=x, t=batched_times, tgt_embs=link, partial_mask=m, A=A, **model_kwargs)

        return torch.clamp_min(x, 0.)
    
    def fast_TME(self, shape, A, link, masks, clip_denoised=True, model_kwargs=None):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = torch.randn(shape, device = device)
        m = (masks>0).reshape(masks.shape).to(device)
        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(x, time_cond, self_cond, clip_x_start=clip_denoised)

            if time_next < 0:
                x = x_start
                continue

            x_start_hat = torch.clamp_min(x_start, 0.)

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x)
            pred_mean = x_start_hat * alpha_next.sqrt() + c * pred_noise
            x = pred_mean + sigma * noise
            x = self.guidance_nt(sample=x, t=time_cond, tgt_embs=link, partial_mask=m, A=A, **model_kwargs)

        return torch.clamp_min(x, 0.)
    
    def guidance_tmc(
        self,
        partial_mask,
        tgt_embs,
        learning_rate,
        sample,
        t
    ):
    
        if t[0].item() < self.num_timesteps * 0.05:
            K = 0
        elif t[0].item() > self.num_timesteps * 0.9:
            K = 3
        elif t[0].item() > self.num_timesteps * 0.75:
            K = 2
            learning_rate = learning_rate * 0.5
        else:
            K = 1
            learning_rate = learning_rate * 0.25

        input_embs_param = torch.nn.Parameter(sample)

        with torch.enable_grad():
            for i in range(K):
                optimizer = torch.optim.Adagrad([input_embs_param], lr=learning_rate)
                optimizer.zero_grad()

                preds = self.model_predictions(x=input_embs_param, t=t, clip_x_start=True)
                traffic_start = preds.pred_x_start

                infill_loss = torch.abs(traffic_start[partial_mask] - tgt_embs[partial_mask])
                loss = infill_loss.mean(dim=0).sum()
                loss.backward()
                optimizer.step()
                
                input_embs_param = torch.nn.Parameter(input_embs_param.data.detach())

        return sample
    
    def guidance_nt(
        self,
        A,
        tgt_embs,
        partial_mask,
        learning_rate,
        sample,
        t
    ):
    
        if t[0].item() < self.num_timesteps * 0.05:
            K = 0
        elif t[0].item() > self.num_timesteps * 0.9:
            K = 3
        elif t[0].item() > self.num_timesteps * 0.75:
            K = 2
            learning_rate = learning_rate * 0.5
        else:
            K = 1
            learning_rate = learning_rate * 0.25

        input_embs_param = torch.nn.Parameter(sample)

        with torch.enable_grad():
            for i in range(K):
                optimizer = torch.optim.Adagrad([input_embs_param], lr=learning_rate)
                optimizer.zero_grad()

                preds = self.model_predictions(x=input_embs_param, t=t, clip_x_start=True)
                traffic_start = preds.pred_x_start

                rec_loss = torch.abs((traffic_start @ A)[partial_mask] - tgt_embs[partial_mask])
                loss = rec_loss.mean(dim=0).sum()
                loss.backward()
                optimizer.step()
                
                input_embs_param = torch.nn.Parameter(input_embs_param.data.detach())

        return sample
    
    def guidance_both(
        self,
        A,
        coef,
        tgt_embs_link,
        partial_mask_link,
        tgt_embs_flow,
        partial_mask_flow,
        learning_rate,
        sample,
        t
    ):
    
        if t[0].item() < self.num_timesteps * 0.05:
            K = 0
        elif t[0].item() > self.num_timesteps * 0.9:
            K = 3
        elif t[0].item() > self.num_timesteps * 0.75:
            K = 2
            learning_rate = learning_rate * 0.5
        else:
            K = 1
            learning_rate = learning_rate * 0.25

        input_embs_param = torch.nn.Parameter(sample)

        with torch.enable_grad():
            for i in range(K):
                optimizer = torch.optim.Adagrad([input_embs_param], lr=learning_rate)
                optimizer.zero_grad()

                preds = self.model_predictions(x=input_embs_param, t=t, clip_x_start=True)
                traffic_start = preds.pred_x_start

                rec_loss_nt = torch.abs((traffic_start @ A)[partial_mask_link] - tgt_embs_link[partial_mask_link])
                rec_loss_ob = torch.abs(traffic_start[partial_mask_flow] - tgt_embs_flow[partial_mask_flow])
                loss = coef * rec_loss_nt.mean(dim=0).sum() + rec_loss_ob.mean(dim=0).sum()
                loss.backward()
                optimizer.step()
                
                input_embs_param = torch.nn.Parameter(input_embs_param.data.detach())

        return sample

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, mask, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        model_out = self.model(x, t, x_self_cond)
        return model_out

    def forward(self, traffic, mask=None, *args, **kwargs):
        b, device, = traffic.shape[0], traffic.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(traffic, mask, t, *args, **kwargs)


if __name__ == "__main__":
    pass