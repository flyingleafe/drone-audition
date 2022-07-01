import sys
import jax
import torch
import optax
import elegy as eg
import haiku as hk

from typing import Any, Sequence, Sized
from jax import numpy as jnp, vmap
from torch.utils.data import random_split, Dataset

sys.path.append("..")

from drone_audition.models import HarmonicNoiseGen
from drone_audition.losses import spectral_mse, log_spectral_distance, mk_elegy_loss
from drone_audition.datasets.dregon import DregonDataset
from drone_audition.datasets.utils import (
    ChannelSelect,
    SamplesSet,
    ds_map,
)
from env import settings


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


def main() -> None:
    EPOCHS = 500
    MODEL_DIR = f"{settings.WORKSPACE_DIR}/models/HarmonicNoiseGen"
    LOGS_DIR = f"{settings.WORKSPACE_DIR}/logs/HarmonicNoiseGen"
    SEED = 42
    BATCH_SIZE = 32

    rseq = hk.PRNGSequence(SEED)

    dregon = DregonDataset(
        data_dir=settings.DREGON_PATH, sample_rate=settings.SAMPLE_RATE
    )

    dregon_3ch = ChannelSelect(dregon, 3)
    samples = SamplesSet(dregon_3ch, 16384)
    train_ds, val_ds = train_val_split(next(rseq), samples, val_pct=0.1)

    train_ds = ds_map(train_ds, lambda x: (x["motor_speed"], x["wav"]))
    val_ds = ds_map(val_ds, lambda x: (x["motor_speed"], x["wav"]))

    train_dl = eg.data.DataLoader(train_ds, BATCH_SIZE, n_workers=4, shuffle=True)
    val_dl = eg.data.DataLoader(val_ds, 1, n_workers=4)

    # prepare model (with batching)
    module = hk.transform_with_state(
        lambda rps: vmap(HarmonicNoiseGen(num_harmonics=50))(rps)
    )
    mse_loss = mk_elegy_loss(spectral_mse)
    log_loss = mk_elegy_loss(log_spectral_distance)

    model = eg.Model(
        module=module,
        loss=[
            mse_loss(),
            log_loss(),
        ],
        optimizer=optax.adam(1e-3),
    )

    model.summary(jnp.array([train_ds[0][0], train_ds[1][0], train_ds[2][0]]))  # type: ignore

    model.fit(
        inputs=train_dl,
        epochs=EPOCHS,
        validation_data=val_dl,
        callbacks=[
            eg.callbacks.ModelCheckpoint(MODEL_DIR, save_best_only=True),
            eg.callbacks.TensorBoard(LOGS_DIR),
        ],
    )

    eval_res = model.evaluate(val_dl)
    print(eval_res)


if __name__ == "__main__":
    main()
