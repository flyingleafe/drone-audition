import torch
from typing import List, Union
from torch import nn

from drone_audition.dsp import freq_series_to_wav
from env import settings

EPS = 1e-8

class PolynomialRegression(nn.Module):
    def __init__(self, max_degree: int, bias=True):
        super().__init__()
        self.poly_coeffs = nn.Parameter(torch.abs(torch.randn(max_degree)*0.0001))
        self.bias = nn.Parameter(torch.randn(1)) if bias else None
        
    def forward(self, x):
        inp = torch.stack([x**i for i in torch.arange(1, self.poly_coeffs.shape[0] + 1)], axis=-1)
        out = torch.matmul(inp, self.poly_coeffs.unsqueeze(-1)).squeeze(-1)
        if self.bias is not None:
            out += self.bias
        return out
    
class PolyWithExpLog(nn.Module):
    def __init__(self, max_degree: int, bias=True):
        super().__init__()
        self.poly = PolynomialRegression(max_degree, bias)
        # log(wx + b) = log(w) + log(x + b/w), so weight inside is not needed
        self.log_logb = nn.Parameter(torch.abs(torch.randn(1)*0.0001))
        self.log_a = nn.Parameter(torch.abs(torch.randn(1)*0.0001))
        # exp(wx + b) = exp(wx)*exp(b)
        self.exp_w = nn.Parameter(torch.abs(torch.randn(1)*0.0001))
        self.exp_a = nn.Parameter(torch.abs(torch.randn(1)*0.0001))
        
    def forward(self, x):
        return self.poly(x) + self.exp_a * torch.exp(self.exp_w * x) + self.log_a * torch.log(x + torch.exp(self.log_logb) + EPS)


class PropellerNoiseGen(nn.Module):
    def __init__(self, use_basis_gain=True, n_harmonics: int = 50, n_blades: int = 2, sample_rate: int = settings.SAMPLE_RATE):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_harmonics = n_harmonics
        self.basis_gain_function = PolynomialRegression(2, bias=False) if use_basis_gain else None
        # self.harmonic_gain_corrections = nn.Parameter(
        #     torch.log(1 / torch.arange(1, n_harmonics + 1)) + torch.randn(n_harmonics)
        # )
        self.harmonic_gain_corrections = nn.Parameter(torch.randn(n_harmonics))
        self.n_blades = n_blades

    def forward(self, speed_rps: torch.Tensor, phase_shift: torch.Tensor):
        assert speed_rps.shape[:-1] == phase_shift.shape
        assert phase_shift.device == phase_shift.device and phase_shift.dtype == phase_shift.dtype

        # make phase shifts for each of the harmonics of BPF from rotor phase shift
        # make it here so that we can probably add per-harmonic learned constant
        # corrections to each of the phase shifts
        coeffs = torch.arange(1, self.n_harmonics + 1).to(speed_rps)
        harmonic_phase_shifts = torch.matmul(
            (phase_shift * float(self.n_blades)).unsqueeze(-1), coeffs.unsqueeze(0)
        )
        
        harmonic_freqs = torch.matmul(
            coeffs.unsqueeze(-1),
            (speed_rps * float(self.n_blades)).unsqueeze(-2)
        )

        harmonics = freq_series_to_wav(harmonic_freqs, harmonic_phase_shifts, sr=self.sample_rate)
        if self.basis_gain_function is not None:
            basis_gains = self.basis_gain_function(speed_rps / 343.0)   # don't ask, it's speed of sound
            gains = basis_gains.unsqueeze(-2).tile((self.n_harmonics, 1)) + \
                self.harmonic_gain_corrections.unsqueeze(-1).tile((speed_rps.shape[-1],))
        else:
            gains = self.harmonic_gain_corrections.unsqueeze(-1).tile((speed_rps.shape[-1],))
        
        return ((torch.exp(gains) + EPS) * harmonics).sum(axis=-2)


class DroneNoiseGen(nn.Module):
    def __init__(
        self,
        use_basis_gain: bool = True,
        n_motors: int = 4,
        n_blades: int = 2,
        n_harmonics: int = 50,
        train_phase_shifts=False,
        sample_rate: int = settings.SAMPLE_RATE
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_motors = n_motors
        self.propeller = PropellerNoiseGen(use_basis_gain, n_harmonics, n_blades, sample_rate)
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
