import jax
import haiku as hk
import numpy as np

from jax import numpy as jnp, vmap
from drone_audition.dsp import rps_to_bpf_harmonics

# Weird error here with unability to subclass the `Callable` from mypy
class LogHarmonicWeightsInitializer(hk.initializers.Initializer):  # type: ignore
    def __init__(self, noise_std=1.0):
        self.noise_std = noise_std

    def __call__(self, shape, dtype):
        k = shape[-1]
        h_weights = (1 / jnp.arange(1, k + 1)).astype(dtype)
        tiled = jnp.tile(h_weights, list(shape[:-1]) + [1])
        noise = self.noise_std * jax.random.normal(hk.next_rng_key(), shape, dtype)
        return jnp.log(tiled) + noise


class HarmonicNoiseGen(hk.Module):
    def __init__(self, num_harmonics, name=None):
        super().__init__(name=name)
        self.num_harmonics = num_harmonics

    def __call__(self, rps):
        harmonics = vmap(rps_to_bpf_harmonics, (0, None))(rps, self.num_harmonics)
        m, k, t = harmonics.shape
        W_init = LogHarmonicWeightsInitializer(1.0 / np.sqrt(k))
        W = hk.get_parameter("W", shape=[m, k], dtype=harmonics.dtype, init=W_init)
        scale = hk.get_parameter(
            "alpha", shape=[], dtype=harmonics.dtype, init=jnp.ones
        )
        return scale * jnp.sum(vmap(jnp.dot)(jnp.exp(W), harmonics), axis=0)
