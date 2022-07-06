from jax import numpy as jnp, scipy as jsp, jit, vmap
from functools import partial
from typing import Optional
from env import settings


@partial(jit, static_argnames=["n_fft", "hop_length"])
def stft(wav, *, n_fft: int = 2048, hop_length: Optional[int] = None):
    """
    JAX implementation of stft with Librosa interface (would expand as needed)
    """
    if hop_length is None:
        hop_length = n_fft // 4

    _, _, S = jsp.signal.stft(  # type: ignore
        wav, fs=settings.SAMPLE_RATE, nperseg=n_fft, noverlap=(n_fft - hop_length)
    )

    return S


@partial(jit, static_argnames=["blades_per_propeller", "sr"])
def rps_to_bpf_phase(rps, blades_per_propeller=2, sr=settings.SAMPLE_RATE):
    bpf = rps * blades_per_propeller
    phase_diff = bpf * 2 * jnp.pi / sr
    unwrapped_phase = jnp.cumsum(phase_diff)
    return unwrapped_phase % (2 * jnp.pi)


def rps_to_bpf_wav(rps, phase_shift=0, sr=settings.SAMPLE_RATE):
    return jnp.sin(rps_to_bpf_phase(rps, sr=sr) + phase_shift)


@partial(jit, static_argnames=["num_harmonics"])
def rps_to_bpf_harmonics(rps, num_harmonics, **kwargs):
    coeffs = jnp.expand_dims(jnp.arange(1, num_harmonics + 1), axis=1)
    rpss = jnp.dot(coeffs, jnp.expand_dims(rps, axis=0))
    return vmap(rps_to_bpf_wav)(rpss)


@partial(jit, static_argnames=["num_harmonics"])
def rps_to_harmonic_sum(rps, num_harmonics=50, **kwargs):
    harmonics = rps_to_bpf_harmonics(rps, num_harmonics, **kwargs)
    coeffs = 1 / jnp.arange(1, num_harmonics + 1)
    return jnp.dot(jnp.expand_dims(coeffs, 0), harmonics).flatten()
