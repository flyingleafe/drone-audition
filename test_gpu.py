import jax
import torch
from jax.lib import xla_bridge

print(jax.default_backend())
print(jax.devices())
print(xla_bridge.get_backend().platform)

print(torch.cuda.is_available())
print(torch.cuda.device_count())