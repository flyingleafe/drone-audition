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


def _slice_last_dim(x: np.ndarray, k: slice) -> np.ndarray:
    slc = [slice(None)] * (x.ndim - 1) + [k]
    return x[tuple(slc)]


def _pad_last_dim(x: np.ndarray, padding, mode: str) -> np.ndarray:
    pd = tuple([(0, 0)] * (x.ndim - 1) + [padding])
    return np.pad(x, pd if len(pd) > 1 else pd[0], mode=mode)  # type: ignore


class SimpleDregonItem(dict):
    def __getitem__(self, key: str | slice):
        if isinstance(key, slice):
            return SimpleDregonItem(
                **{
                    "path": self["path"],
                    "wav": _slice_last_dim(self["wav"], key),
                    "ts": _slice_last_dim(self["ts"], key),
                    "motor_speed": _slice_last_dim(self["motor_speed"], key),
                    "angular_velocity": _slice_last_dim(self["angular_velocity"], key),
                    "acceleration": _slice_last_dim(self["acceleration"], key),
                }
            )

        return super().__getitem__(key)

    def __len__(self) -> int:
        return len(self["ts"])

    def pad(self, padding, mode) -> SimpleDregonItem:
        return SimpleDregonItem(
            **{
                "path": self["path"],
                "wav": _pad_last_dim(self["wav"], padding, mode),
                "ts": _pad_last_dim(self["ts"], padding, mode),
                "motor_speed": _pad_last_dim(self["motor_speed"], padding, mode),
                "angular_velocity": _pad_last_dim(
                    self["angular_velocity"], padding, mode
                ),
                "acceleration": _pad_last_dim(self["acceleration"], padding, mode),
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
        coordinates = loadmat(os.path.join(data_dir, "coordinates.mat"))

        self.mic_pos = coordinates["micPos"]
        self.rotors_pos = coordinates["rotorsPos"]

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

        audiots = loadmat(self.audiots[index])["audio_timestamps"].flatten()

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

        motor_speed_int = interp1d(motor_ts, motor_speed, kind="linear")(audiots)
        angular_velocity_int = interp1d(imu_ts, angular_velocity, kind="linear")(
            audiots
        )
        acceleration_int = interp1d(imu_ts, acceleration, kind="linear")(audiots)

        # start from 0 and avoid precision errors when casting to float32
        audiots -= audiots[0]

        return SimpleDregonItem(
            **{
                "path": path,
                "wav": wav,
                "ts": audiots.astype(np.float32),
                "motor_speed": motor_speed_int.astype(np.float32),
                "angular_velocity": angular_velocity_int.astype(np.float32),
                "acceleration": acceleration_int.astype(np.float32),
                "orig_motor_speed": motor_speed.astype(np.float32),
            }
        )
