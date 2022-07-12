import sys
import os
import torch
import numpy as np
import pytorch_lightning as pl

from typing import Tuple
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch import optim
from torch.utils.data import DataLoader
from asteroid import System
from asteroid.losses import SingleSrcMultiScaleSpectral

sys.path.append("..")

from drone_audition.models import DroneNoiseGen
from drone_audition.datasets.dregon import DregonDataset
from drone_audition.datasets.utils import (
    ChannelSelect,
    SamplesSet,
    ds_map,
    train_val_split,
)
from env import settings


def prepare_dataloaders(rng, batch_size) -> Tuple[DataLoader, DataLoader]:
    dregon = DregonDataset(
        data_dir=settings.DREGON_PATH, sample_rate=settings.SAMPLE_RATE
    )

    dregon_3ch = ChannelSelect(dregon, 3)
    samples = SamplesSet(dregon_3ch, 16384)
    train_ds, val_ds = train_val_split(rng, samples, val_pct=0.1)

    train_ds = ds_map(train_ds, lambda x: (x["motor_speed"], x["wav"]))
    val_ds = ds_map(val_ds, lambda x: (x["motor_speed"], x["wav"]))

    train_dl = DataLoader(train_ds, batch_size, num_workers=4, shuffle=True)
    val_dl = DataLoader(val_ds, 1, num_workers=4)

    return train_dl, val_dl


def main() -> None:
    EPOCHS = 500
    MODEL_DIR = f"{settings.WORKSPACE_DIR}/models"
    MODEL_VERSION = "01"
    LOGS_DIR = f"{settings.WORKSPACE_DIR}/logs"
    SEED = 42
    BATCH_SIZE = 32
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    train_dl, val_dl = prepare_dataloaders(SEED, BATCH_SIZE)
    model = DroneNoiseGen()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

    stft_n_ffts = np.array([4096, 2048, 1024, 512])
    loss = SingleSrcMultiScaleSpectral(stft_n_ffts, stft_n_ffts, stft_n_ffts / 4, 0.1)

    system = System(model, optimizer, loss, train_dl, val_dl, scheduler)
    tb_logger = pl.loggers.TensorBoardLogger(
        LOGS_DIR, name="DroneNoiseGen", version=f"v{MODEL_VERSION}"
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", mode="min"),
        ModelCheckpoint(
            filename="{epoch:02d}-{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
            save_top_k=2,
        ),
    ]

    trainer_kwargs = dict(
        max_epochs=EPOCHS, logger=tb_logger, callbacks=callbacks, deterministic=True
    )

    if torch.cuda.is_available():
        trainer_kwargs = {**trainer_kwargs, "devices": 1, "accelerator": "gpu"}

    trainer = pl.Trainer(**trainer_kwargs)  # type: ignore
    trainer.fit(system)

    torch.save(model.state_dict(), f"{MODEL_DIR}/DroneNoiseGen_v{MODEL_VERSION}.pt")


if __name__ == "__main__":
    main()
