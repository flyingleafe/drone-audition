from __future__ import annotations

import os
import bisect
import numpy as np
import librosa as lr

from torch.utils.data import Dataset
from glob import glob
from scipy.io import loadmat
from scipy.interpolate import interp1d

from .utils import cached_in


def _load_audio(path: str | os.PathLike, sr: int = 44100):
    audio, _ = lr.load(path, sr=sr, mono=False, dtype="float32")
    return audio


def _resample_ts(ts: np.ndarray, samples: int) -> np.ndarray:
    return np.linspace(ts[0], ts[-1], samples)


def clean_time_duplicates(ts: np.ndarray, *seqs):
    ix = np.ones(len(ts), dtype=bool)
    ix[1:] = ix[1:] ^ (ts[1:] == ts[:-1])
    return [ts[ix]] + [s[:, ix] for s in seqs]


class SimpleDregonItem(dict):
    def __getitem__(self, key: str | slice):
        if isinstance(key, slice):
            return SimpleDregonItem(
                **{
                    "path": self["path"],
                    "wav": self["wav"][:, key],
                    "ts": self["ts"][key],
                    "motor_speed": self["motor_speed"][:, key],
                    "angular_velocity": self["angular_velocity"][:, key],
                    "acceleration": self["acceleration"][:, key],
                }
            )

        return super().__getitem__(key)

    def __len__(self) -> int:
        return len(self["ts"])

    def pad(self, padding, mode) -> SimpleDregonItem:
        return SimpleDregonItem(
            **{
                "path": self["path"],
                "wav": np.pad(self["wav"], ((0, 0), padding), mode=mode),
                "ts": np.pad(self["ts"], padding, mode=mode),
                "motor_speed": np.pad(
                    self["motor_speed"], ((0, 0), padding), mode=mode
                ),
                "angular_velocity": np.pad(
                    self["angular_velocity"], ((0, 0), padding), mode=mode
                ),
                "acceleration": np.pad(
                    self["acceleration"], ((0, 0), padding), mode=mode
                ),
            }
        )


class DregonDataset(Dataset):
    """
    DREGON dataset
    """

    dataset_name = "DREGON"
    download_url = ""
    default_sample_rate = 44100

    def __init__(
        self,
        data_dir: os.PathLike,
        subset: str = "nosource",
        sample_rate: int = 44100,
        cached: bool = True,
    ) -> None:
        self.data_dir = os.path.join(data_dir, subset)
        self.sample_rate = sample_rate

        self.wavs = glob(self.data_dir + "/*.wav")
        self.motors = [p.replace(".wav", "_motors.mat") for p in self.wavs]
        self.audiots = [p.replace(".wav", "_audiots.mat") for p in self.wavs]
        self.imus = [p.replace(".wav", "_imu.mat") for p in self.wavs]
        self.cache: dict | None = {} if cached else None

    def __len__(self) -> int:
        return len(self.wavs)

    @cached_in("cache")
    def __getitem__(self, index: int) -> SimpleDregonItem:
        path = self.wavs[index]
        wav = _load_audio(path, sr=self.sample_rate)

        audiots = (
            loadmat(self.audiots[index])["audio_timestamps"]
            .astype(np.float32)
            .flatten()
        )

        if self.sample_rate != DregonDataset.default_sample_rate:
            audiots = _resample_ts(audiots, wav.shape[1])

        motor = loadmat(self.motors[index], mat_dtype=False)["motor"][0, 0]
        motor_ts = motor[0].flatten()
        motor_speed = motor[1].T

        imu = loadmat(self.imus[index])["imu"][0, 0]
        imu_ts = imu[0].flatten()
        angular_velocity = imu[1].T
        acceleration = imu[2].T

        motor_ts, motor_speed = clean_time_duplicates(motor_ts, motor_speed)
        imu_ts, angular_velocity, acceleration = clean_time_duplicates(
            imu_ts, angular_velocity, acceleration
        )

        # Cut everything up so that all arrays start and end with same timestamp
        min_ts = (
            np.max([audiots[0], motor_ts[0], imu_ts[0]]) + 6.0
        )  # just cut out the weird start of motor speeds and everyhing else
        max_ts = np.min([audiots[-1], motor_ts[-1], imu_ts[-1]])

        # Cut time quantized by audio samples
        al = bisect.bisect_left(audiots, min_ts)
        ar = bisect.bisect_right(audiots, max_ts)
        audiots = audiots[al:ar]
        wav = wav[:, al:ar]

        motor_speed = interp1d(motor_ts, motor_speed, kind="linear")(audiots)
        angular_velocity = interp1d(imu_ts, angular_velocity, kind="linear")(audiots)
        acceleration = interp1d(imu_ts, acceleration, kind="linear")(audiots)

        # start from 0
        audiots -= audiots[0]

        return SimpleDregonItem(
            **{
                "path": path,
                "wav": wav,
                "ts": audiots.astype(np.float32),
                "motor_speed": motor_speed.astype(np.float32),
                "angular_velocity": angular_velocity.astype(np.float32),
                "acceleration": acceleration.astype(np.float32),
            }
        )
