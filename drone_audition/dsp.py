import torch

from env import settings


def freq_series_to_phase(freq: torch.Tensor, sr: int = settings.SAMPLE_RATE):
    """
    Transform variable frequency signal to wrapped phase
    (starting with phase 0)

    Args:
        freq (torch.Tensor): input frequency signal, shaped [..., T]

    Returns:
        :class:`torch.Tensor` with shape [..., T] - wrapped phases
    """
    phase_diff = freq * 2 * torch.pi / sr
    unwrapped_phase = phase_diff.cumsum(-1)
    return unwrapped_phase % (2 * torch.pi)


def freq_series_to_wav(
    freq: torch.Tensor, phase_shift: torch.Tensor, sr: int = settings.SAMPLE_RATE
):
    """
    Transform variable frequency signal to waveform with given phase shifts.
    Pretty much this is frequency modulation of the signal.

    Args:
        freq (torch.Tensor): input frequency signal, shaped [B_1, .. B_k, T]
        phase_shift (torch.Tensor): phase shift for each of signals, shaped [B_1, .. B_k, N], where N is the number of harmonics

    Returns:
        :class:`torch.Tensor` with shape [N_1, .. N_k, T] - sine waves with variable frequency
    """
    return torch.sin(freq_series_to_phase(freq, sr) + phase_shift.unsqueeze(-1))


def freq_series_to_harmonics(freq: torch.Tensor, phase_shifts: torch.Tensor, **kwargs):
    assert freq.shape[:-1] == phase_shifts.shape[:-1]
    n_harmonics = phase_shifts.shape[-1]
    coeffs = torch.arange(1, n_harmonics + 1).to(freq)
    harmonic_freqs = torch.matmul(coeffs.unsqueeze(-1), freq.unsqueeze(-2))
    return freq_series_to_wav(harmonic_freqs, phase_shifts, **kwargs)


def freq_series_to_harmonic_sum(
    freq: torch.Tensor, phase_shifts: torch.Tensor, **kwargs
):
    harmonics = freq_series_to_harmonics(freq, phase_shifts, **kwargs)
    coeffs = 1 / torch.arange(1, harmonics.shape[-2] + 1).to(freq)
    return torch.matmul(coeffs.unsqueeze(0), harmonics).squeeze(-2)
