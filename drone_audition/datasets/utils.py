import torch
import jax
import torch.nn.functional as F
import numpy as np
import jax.numpy as jnp

from typing import Optional, Sequence, Any, Sized
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from collections import abc


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


def ds_map(ds: Dataset, fn) -> Dataset:
    cls = type(
        ds.__class__.__name__ + "_mapped",
        (Dataset,),
        {
            "__len__": lambda self: len(ds),  # type: ignore
            "__getitem__": lambda self, index: fn(ds[index]),
        },
    )
    return cls()


class ChannelSelect(Dataset):
    """
    Just select a single channel from dataset with wavs
    """

    def __init__(self, ds: Dataset, ch: int, wav_field: str = "wav"):
        self.ds = ds
        self.ch = ch
        self.wav_field = wav_field

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        res = deepcopy(self.ds[index])
        res[self.wav_field] = res[self.wav_field][self.ch]
        return res


class ChannelSplit(Dataset):
    """
    Dataset wrapper which splits the multi-channel wavs into mono wavs, preserving
    all the corresponding metainfo.

    The assumption is that all the audio in the dataset has the same
    number of channels.
    """

    def __init__(
        self, ds: Dataset, wav_field: str = "wav", num_ch: Optional[int] = None
    ):
        self.ds = ds
        self.wav_field = wav_field

        if num_ch is None:
            sample_wav = self.ds[0][self.wav_field]
            self.num_ch = sample_wav.shape[0]
        else:
            self.num_ch = num_ch

    def __len__(self):
        return len(self.ds) * self.num_ch

    def __getitem__(self, index):
        wav_idx = index // self.num_ch
        ch_idx = index % self.num_ch

        res = self.ds[wav_idx].copy()
        wav = res[self.wav_field][ch_idx]
        res[self.wav_field] = wav
        res["channel"] = ch_idx

        return res


class SamplesSet(Dataset):
    """
    Cut the given sequences into a series of samples of a given length.
    """

    def __init__(self, ds, sample_len):
        self.ds = ds
        self.sample_len = sample_len
        self.sample_nums = [0] * len(self.ds)

        for i in tqdm(range(len(self.ds)), "Reading dataset and calculating samples"):
            ln = len(self.ds[i])
            sample_num = ln // self.sample_len
            if ln % self.sample_len > 0:
                sample_num += 1

            self.sample_nums[i] = sample_num

    def __len__(self):
        return sum(self.sample_nums)

    def __getitem__(self, index):
        back_idx = 0
        while back_idx < len(self.sample_nums) and index >= self.sample_nums[back_idx]:
            index -= self.sample_nums[back_idx]
            back_idx += 1

        if back_idx >= len(self.sample_nums):
            raise IndexError("out of range")

        origin = self.ds[back_idx]
        l = index * self.sample_len
        r = l + self.sample_len
        chunk = origin[l:r]
        if len(chunk) < self.sample_len:
            chunk = chunk.pad((0, self.sample_len - len(chunk)), mode="reflect")

        return chunk


class SizedDataset(Dataset, Sized):
    ...


def train_val_split(
    rng: jnp.ndarray, ds: SizedDataset, val_pct: float = 0.2
) -> Sequence[Dataset[Any]]:
    assert val_pct < 1.0
    val_len = int(len(ds) * val_pct)
    train_len = len(ds) - val_len
    return random_split(
        ds,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(
            int(jax.random.randint(rng, [], 0, 99999999))
        ),
    )


def numpy_collate(batch):
    """
    A simplified rewrite of default_collate from Pytorch which does not use torch tensors.
    """
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, np.ndarray):
        return np.stack(batch)
    elif isinstance(elem, (str, bytes)):
        return batch
    elif isinstance(elem, abc.Sequence):
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(e) == elem_size for e in it):
            raise RuntimeError("each element list of batch should be of equal size")

        transposed = list(zip(*batch))
        try:
            return elem_type([numpy_collate(samples) for samples in transposed])
        except TypeError:
            return [numpy_collate(samples) for samples in transposed]
    elif isinstance(elem, abc.Mapping):
        try:
            return elem_type({k: numpy_collate([d[k] for d in batch]) for k in elem})
        except TypeError:
            return {k: numpy_collate([d[k] for d in batch]) for k in elem}
    else:
        return np.array(batch)


class NumpyLoader(DataLoader):
    """
    Dataloader which does not turn numpy arrays into torch tensors, stolen from Jax tutorial
    Uses `numpy_collate` as collate function.
    """

    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )
