import torch
import sys
import os
import numpy as np
import soundfile as sf
import jax.numpy as jnp
import matplotlib.pyplot as plt
import librosa as lr
import librosa.display as lrd
import IPython.display as ipd

sys.path.append("..")

from env import settings


def save_wav(wav, name: str):
    samples_dir = os.path.join(settings.WORKSPACE_DIR, "wav_samples")
    os.makedirs(samples_dir, exist_ok=True)
    sf.write(os.path.join(samples_dir, f"{name}.wav"), wav, settings.SAMPLE_RATE)


def plot_dwim(*args, **kwargs):
    """
    Function for plotting whatever. Functionality is going to be updated.
    """
    if len(args) == 1:
        y = args[0]
        plot_dwim(np.arange(y.shape[-1]), y, **kwargs)
    else:
        x, y = args[0], args[1]
        sh = y.shape
        if len(sh) == 1:
            plt.plot(x, y, **kwargs)
        else:
            for i in range(sh[0]):
                plot_dwim(x, y[i], alpha=0.5, label=f"ch{i}")
        plt.legend()


def show_wav(
    wav,
    sr=settings.SAMPLE_RATE,
    figsize=(10, 4),
    specgram_lib="librosa",
    save_to=None,
    n_fft=2048,
    hop_length=None,
    align="horizontal",
):

    if isinstance(wav, dict):
        if "wav" in wav or "audio" in wav:
            wav = wav.get("wav", wav["audio"])
            return show_wav(
                wav,
                sr=sr,
                figsize=figsize,
                specgram_lib=specgram_lib,
                save_to=save_to,
                n_fft=n_fft,
                hop_length=hop_length,
            )
        else:
            raise KeyError("No keys 'wav' or 'audio' in a dict")

    if hop_length is None:
        hop_length = n_fft // 4

    if type(wav) == str:
        wav, sr = lr.load(wav)
    elif type(wav) == torch.Tensor:
        wav = wav.detach().numpy()
    elif isinstance(wav, jnp.DeviceArray):
        wav = np.array(wav)

    if len(wav.shape) > 1:
        if wav.shape[0] > 1:
            print(f"Showing only channel 0 (out of {wav.shape[0]} channels)")
        wav = wav[0]  # take only first channel

    if align == "horizontal":
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    elif align == "vertical":
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(figsize[1], figsize[0]))

    plt.sca(axes[0])
    lrd.waveshow(wav, sr=sr, ax=axes[0])
    plt.ylabel("Amplitude")

    tticks, tlabels = plt.xticks()

    plt.sca(axes[1])
    if specgram_lib == "librosa":
        S_db = lr.amplitude_to_db(
            np.abs(lr.stft(wav, n_fft=n_fft, hop_length=hop_length))
        )
        img = lrd.specshow(
            S_db,
            sr=sr,
            ax=axes[1],
            hop_length=hop_length,
            x_axis="s",
            y_axis="hz",
            auto_aspect=False,
        )
        fig.colorbar(img, ax=axes[1])
    elif specgram_lib == "matplotlib":
        plt.specgram(wav, Fs=sr, mode="magnitude")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
    else:
        raise ValueError(
            f"Invalid `specgram_lib={specgram_lib}`, should be one of (`librosa`, `matplotlib`)"
        )

    plt.xticks(tticks)

    fig.tight_layout()
    if save_to is not None:
        plt.savefig(save_to, bbox_inches="tight")

    plt.show()

    ipd.display(ipd.Audio(wav, rate=sr))


def plt_wav_speeds(item, figsize=(15, 10)):
    fig, axes = plt.subplots(figsize=figsize, nrows=3, ncols=1, sharex=True)
    plt.sca(axes[0])
    wav = item["wav"]
    if len(wav.shape) > 1:
        wav = wav[0, :]

    plt.plot(item["ts"], wav)
    plt.ylabel("Amplitude")

    plt.sca(axes[1])
    for i in range(4):
        plt.plot(item["ts"].flatten(), item["motor_speed"][i], label=f"m{i}", alpha=0.5)

    plt.ylabel("Motor speed")
    plt.xlabel("Time")
    plt.legend()

    plt.sca(axes[2])
    plt.plot(
        item["ts"].flatten(),
        item["angular_velocity"][2],
        label="ang.velocity (z-axis)",
        alpha=0.5,
    )
    plt.plot(
        item["ts"].flatten(),
        item["acceleration"][2],
        label="acceleration (z-axis)",
        alpha=0.5,
    )
    plt.ylabel("IMU measurements")
    plt.xlabel("Time")
    plt.legend()


def plot_avg_mag(wav, log=False, **kwargs):
    avg_mag = jnp.mean(jnp.abs(lr.stft(np.array(wav), center=False, **kwargs)), axis=-1)
    if log:
        avg_mag = jnp.log(avg_mag)
    return plt.plot(jnp.arange(len(avg_mag)), avg_mag)


def plot_avg_mag_with_log(wav, **kwargs):
    fig, axes = plt.subplots(figsize=(10, 4), nrows=1, ncols=2)
    plt.sca(axes[0])
    plot_avg_mag(wav, **kwargs)
    plt.sca(axes[1])
    plot_avg_mag(wav, log=True, **kwargs)
    fig.tight_layout()
