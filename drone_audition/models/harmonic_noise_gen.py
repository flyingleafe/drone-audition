import torch
from torch import nn

from ..dsp import freq_series_to_harmonics


class PropellerNoiseGen(nn.Module):
    def __init__(self, n_harmonics: int = 50, n_blades: int = 2):
        super().__init__()
        self.log_A = nn.Parameter(torch.randn(n_harmonics))
        self.phi = nn.Parameter(torch.randn(n_harmonics))
        self.n_blades = n_blades

    def forward(self, speed_rps: torch.Tensor):
        harmonics = freq_series_to_harmonics(
            speed_rps * float(self.n_blades),
            self.phi.tile(list(speed_rps.shape[:-1]) + [1]),
        )

        return torch.matmul(torch.exp(self.log_A.unsqueeze(0)), harmonics)


class DroneNoiseGen(nn.Module):
    def __init__(self, n_motors: int = 4, n_blades: int = 2, n_harmonics: int = 50):
        super().__init__()
        self.propellers = nn.ModuleList(
            [PropellerNoiseGen(n_harmonics, n_blades) for i in range(n_motors)]
        )

        # no exponentiation, they have no reason to get negatve right
        self.motor_coeffs = nn.Parameter(torch.rand(n_motors) + 1.0)

    def forward(self, speed_rps: torch.Tensor):
        assert speed_rps.shape[-2] == len(self.propellers)

        propeller_outputs = []
        for i in range(len(self.propellers)):
            inp = speed_rps.index_select(
                -2, torch.tensor(i).to(device=speed_rps.device)
            ).squeeze(-2)
            outp = self.propellers[i](inp)
            propeller_outputs.append(outp)

        stacked = torch.stack(propeller_outputs, -2)
        return torch.matmul(self.motor_coeffs.unsqueeze(0), stacked)
