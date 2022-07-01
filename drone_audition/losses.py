import elegy as el
from jax import numpy as jnp, jit
from .dsp import stft

EPS = 1e-8


def mk_elegy_loss(fn):
    ps_case_name = fn.__name__.replace("_", " ").title().replace(" ", "")
    return type(
        ps_case_name,
        (el.Loss,),
        {
            # TODO: support varargs, but preserve the method signature?
            "call": lambda self, target, preds: fn(target, preds),
        },
    )


@jit
def spectral_mse(target, preds):
    mag_e = jnp.abs(stft(preds))
    mag_t = jnp.abs(stft(target))
    return jnp.mean((mag_e - mag_t) ** 2)


@jit
def log_spectral_distance(target, preds):
    mag_e = jnp.abs(stft(preds))
    mag_t = jnp.abs(stft(target))
    return jnp.mean((jnp.log(mag_e + EPS) - jnp.log(mag_t + EPS)) ** 2)
