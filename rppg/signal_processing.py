"""
CHROM rPPG algorithm (de Haan & Jeanne, IEEE TBME 2013)
given a (N, 3) array of mean RGB traces and the sampling rate,
returns the rPPG signal and estimated BPM + respiratory rate
"""
import numpy as np
from scipy.signal import butter, filtfilt

def _bandpass(sig: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    nyq = fs / 2.0
    b, a = butter(4, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, sig)

def chrom(rgb: np.ndarray, fs: float) -> np.ndarray:
    # CHROM algorithm -> rPPG signal
    # normalise each channel by its mean
    rn = rgb[:, 0] / (rgb[:, 0].mean() + 1e-9)
    gn = rgb[:, 1] / (rgb[:, 1].mean() + 1e-9)
    bn = rgb[:, 2] / (rgb[:, 2].mean() + 1e-9)

    x = 3 * rn - 2 * gn # chrominance X
    y = 1.5 * rn + gn - 1.5 * bn  # chrominance Y

    # bandpass 0.67–4 Hz (40–240 BPM)
    xf = _bandpass(x, fs, 0.67, 4.0)
    yf = _bandpass(y, fs, 0.67, 4.0)

    alpha = xf.std() / (yf.std() + 1e-9)
    return xf - alpha * yf

def estimate_bpm(signal: np.ndarray, fs: float,
                 lo_hz: float = 0.67, hi_hz: float = 4.0) -> float:
    # dominant frequency via zero-padded FFT -> BPM (finer resolution than Welch)
    # zero-pad to 8x length for sub-0.1 Hz resolution even on 5-s chunks
    n_fft = max(len(signal) * 8, int(fs * 60))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    psd = np.abs(np.fft.rfft(signal - signal.mean(), n=n_fft)) ** 2
    mask = (freqs >= lo_hz) & (freqs <= hi_hz)
    peak_hz = freqs[mask][np.argmax(psd[mask])]
    return peak_hz * 60.0

def estimate_rr(rgb: np.ndarray, fs: float) -> float:
    """
    respiratory rate from the green-channel baseline drift (0.1-0.5 Hz).
    breathing modulates skin blood volume at a lower frequency than HR.
    we low-pass the normalised green channel and find the dominant frequency.
    requires >=10 s of signal for reliable estimation; returns nan otherwise.
    """
    min_samples = int(fs * 10)
    if len(rgb) < min_samples:
        return float("nan")
    g = rgb[:, 1] / (rgb[:, 1].mean() + 1e-9) - 1.0  # zero-mean normalised green
    # low-pass at 0.6 Hz to isolate respiratory band
    nyq = fs / 2.0
    b, a = butter(3, 0.6 / nyq, btype="low")
    g_lp = filtfilt(b, a, g)
    n_fft = max(len(g_lp) * 8, int(fs * 60))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    psd   = np.abs(np.fft.rfft(g_lp - g_lp.mean(), n=n_fft)) ** 2
    mask  = (freqs >= 0.1) & (freqs <= 0.5)
    if not mask.any():
        return float("nan")
    peak_hz = freqs[mask][np.argmax(psd[mask])]
    return peak_hz * 60.0
