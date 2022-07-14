import torch
from torch import nn

from ..dsp import freq_series_to_harmonics


class PropellerNoiseGen(nn.Module):
    def __init__(self, n_harmonics: int = 50, n_blades: int = 2):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.log_A = nn.Parameter(
            torch.log(1 / torch.arange(1, n_harmonics + 1)) + torch.randn(n_harmonics)
        )
        self.n_blades = n_blades

    def forward(self, speed_rps: torch.Tensor, phase_shift: torch.Tensor):
        assert speed_rps.shape[:-1] == phase_shift.shape

        # make phase shifts for each of the harmonics of BPF from rotor phase shift
        # make it here so that we can probably add per-harmonic learned constant
        # corrections to each of the phase shifts
        coeffs = torch.arange(1, self.n_harmonics + 1).to(phase_shift)
        harmonic_phase_shifts = torch.matmul(
            (phase_shift * float(self.n_blades)).unsqueeze(-1), coeffs.unsqueeze(0)
        )

        harmonics = freq_series_to_harmonics(
            speed_rps * float(self.n_blades),
            harmonic_phase_shifts,
        )

        return torch.matmul(torch.exp(self.log_A.unsqueeze(0)), harmonics).squeeze(-2)


class DroneNoiseGen(nn.Module):
    def __init__(
        self,
        n_motors: int = 4,
        n_blades: int = 2,
        n_harmonics: int = 50,
        train_phase_shifts=False,
    ):
        super().__init__()
        self.n_motors = n_motors
        self.propeller = PropellerNoiseGen(n_harmonics, n_blades)
        self.log_motor_coeffs = nn.Parameter(torch.rand(n_motors))
        self.phase_shifts = nn.Parameter(
            torch.zeros(n_motors), requires_grad=train_phase_shifts
        )

    def forward(self, speed_rps: torch.Tensor):
        assert speed_rps.shape[-2] == self.n_motors

        propeller_outputs = self.propeller(
            speed_rps, self.phase_shifts.tile(list(speed_rps.shape[:-2]) + [1])
        )

        return torch.matmul(
            torch.exp(self.log_motor_coeffs.unsqueeze(0)), propeller_outputs
        ).squeeze(-2)
