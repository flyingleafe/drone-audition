import torch
import torch.nn.functional as F
import numpy as np
import jax.numpy as jnp

from torch.utils.data._utils.collate import default_collate


def cached_in(var_name: str):
    """
    Decorator function which makes the __getitem__ results cached
    """

    def decorated(func):
        def inner(self, index):
            cache = getattr(self, var_name, None)
            if cache is not None and index in cache:
                return cache[index]

            res = func(self, index)

            if cache is not None:
                cache[index] = res

            return res

        return inner

    return decorated


def normalize_volume(wav):
    if isinstance(wav, torch.Tensor):
        max_amp = torch.max(torch.abs(wav))
    else:
        max_amp = jnp.max(jnp.abs(wav))

    coeff = 0.95 / max_amp
    return wav * coeff


def online_mixing_collate(batch):
    """Mix target sources to create new mixtures.
    Output of the default collate function is expected to return two objects:
    inputs and targets.
    """
    # Inputs (batch, time) / targets (batch, n_src, time)
    inputs, targets = default_collate(batch)
    batch, n_src, _ = targets.shape

    energies = torch.sum(targets**2, dim=-1, keepdim=True)
    new_src = []
    for i in range(targets.shape[1]):
        new_s = targets[torch.randperm(batch), i, :]
        new_s = new_s * torch.sqrt(energies[:, i] / (new_s**2).sum(-1, keepdims=True))
        new_src.append(new_s)

    targets = torch.stack(new_src, dim=1)
    inputs = targets.sum(1)
    return inputs, targets


def cal_adjusted_rms(clean_rms, snr):
    a = float(snr) / 20
    noise_rms = clean_rms / (10**a)
    return noise_rms


def cal_rms(amp):
    return np.sqrt(np.mean(np.square(amp), axis=-1))


def add_noise_with_snr(clean, noise, snr):
    clean_rms = cal_rms(clean)
    noise_rms = cal_rms(noise)
    adjusted_noise_rms = cal_adjusted_rms(clean_rms, snr)

    noise = noise * (adjusted_noise_rms / noise_rms)

    mixed = clean + noise
    alpha = 1.0

    if mixed.max(axis=0) > 1 or mixed.min(axis=0) < -1:
        if mixed.max(axis=0) >= abs(mixed.min(axis=0)):
            alpha = 1.0 / mixed.max(axis=0)
        else:
            alpha = -1.0 / mixed.min(axis=0)
        mixed = mixed * alpha

    return mixed


def cut_or_pad(tensor, length):
    if tensor.shape[0] >= length:
        return tensor[:length]
    else:
        if isinstance(tensor, torch.Tensor):
            return F.pad(tensor, (0, length - tensor.shape[0]))
        else:
            return np.pad(tensor, ((0, length - tensor.shape[0]),))


def crop_or_wrap(wav, crop_len, offset):
    if len(wav) < crop_len:
        n_repeat = int(np.ceil(float(crop_len) / float(len(wav))))
        wav_ex = np.tile(wav, n_repeat)
        wav = wav_ex[0:crop_len]
    else:
        offset = offset % (len(wav) - crop_len + 1)
        wav = wav[offset : offset + crop_len]
    return wav
