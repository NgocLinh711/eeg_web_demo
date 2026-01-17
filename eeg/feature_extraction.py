# eeg/feature_extraction.py
import numpy as np
from scipy.signal import welch, csd, coherence

# ==========================================
# CONSTANTS (COPY TỪ NOTEBOOK CỦA BẠN)
# ==========================================
FS = 500.0

# PSD config
PSD_FMIN = 1.0
PSD_FMAX = 45.0
LOG_PSD  = True
EPS      = 1e-12
WELCH_WIN_SEC = 2.0
WELCH_OVERLAP = 0.5

# Coherence config: tensor [epochs, bands, ch, ch]
COH_BANDS = [
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 45.0),
]
COH_DTYPE = np.float16

def welch_params(fs, n_samples):
    nperseg = int(round(WELCH_WIN_SEC * fs))
    nperseg = min(max(nperseg, 16), n_samples)
    noverlap = int(round(nperseg * WELCH_OVERLAP))
    noverlap = min(noverlap, nperseg - 1)
    return nperseg, noverlap

def compute_psd_epoch(epoch_chxT, fs=FS):
    """Tính PSD cho 1 epoch"""
    nperseg, noverlap = welch_params(fs, epoch_chxT.shape[-1])
    
    freqs, pxx = welch(
        epoch_chxT,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        axis=-1,
        detrend="constant",
        scaling="density",
    )
    
    keep = (freqs >= PSD_FMIN) & (freqs <= PSD_FMAX)
    # freqs = freqs[keep] # Không cần return freqs
    pxx = pxx[:, keep]
    
    if LOG_PSD:
        pxx = np.log(pxx + EPS)
        
    return pxx.astype(np.float32)

def compute_coh_epoch(epoch_chxT, fs, bands):
    """
    Tính coherence bằng scipy.signal.coherence (từng cặp kênh)
    - Sử dụng scaling='spectrum' (không density) để tránh lỗi normalize
    - nperseg nhỏ phù hợp epoch 2s
    """
    ch, nT = epoch_chxT.shape
    nperseg = min(256, nT)  # 256 điểm FFT cho Fs=500, epoch 2s

    coh = np.full((len(bands), ch, ch), 1.0, dtype=COH_DTYPE)

    for i in range(ch):
        for j in range(i + 1, ch):
            f, cxy = coherence(
                epoch_chxT[i],
                epoch_chxT[j],
                fs=fs,
                nperseg=nperseg,
                noverlap=nperseg // 2,
                window='hann',
                detrend='constant',
            )

            for b, (_, f_low, f_high) in enumerate(bands):
                idx = np.where((f >= f_low) & (f <= f_high))[0]
                if idx.size > 0:
                    mean_coh = np.mean(cxy[idx])
                    coh[b, i, j] = mean_coh
                    coh[b, j, i] = mean_coh

    return coh