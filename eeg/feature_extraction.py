# eeg/feature_extraction.py
import numpy as np
from scipy.signal import welch, csd

# ==========================================
# CONSTANTS (COPY TỪ NOTEBOOK CỦA BẠN)
# ==========================================
FS = 500.0
WELCH_WIN_SEC = 2.0
WELCH_OVERLAP = 0.5
PSD_FMIN = 1.0
PSD_FMAX = 45.0
LOG_PSD = True
EPS = 1e-12

COH_BANDS = [
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 45.0),
]

def welch_params(fs, n_samples):
    """Logic tính nperseg/noverlap gốc từ notebook"""
    nperseg = int(round(WELCH_WIN_SEC * fs))
    # nperseg = min(max(nperseg, 16), n_samples) # Dòng này trong notebook phòng hờ epoch ngắn
    noverlap = int(round(nperseg * WELCH_OVERLAP))
    # noverlap = min(noverlap, nperseg - 1)
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

def compute_coh_epoch(epoch_chxT, fs=FS):
    """Tính Coherence cho 1 epoch (Logic CSD -> Average Bands)"""
    ch, nT = epoch_chxT.shape
    nperseg, noverlap = welch_params(fs, nT)

    # 1. Tính mẫu số (Auto Power Spectra)
    freqs, Sxx = welch(
        epoch_chxT,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        axis=-1,
        detrend="constant",
        scaling="density",
        return_onesided=True,
    )

    # Pre-calculate band indices
    band_idxs = []
    for _, f1, f2 in COH_BANDS:
        idx = np.where((freqs >= f1) & (freqs <= f2))[0]
        band_idxs.append(idx)

    B = len(COH_BANDS)
    coh = np.zeros((B, ch, ch), dtype=np.float32)
    
    # Fill diagonal
    for b in range(B):
        np.fill_diagonal(coh[b], 1.0)

    # 2. Tính Cross Spectra từng cặp
    for i in range(ch):
        xi = epoch_chxT[i]
        for j in range(i + 1, ch):
            xj = epoch_chxT[j]
            _, Sxy = csd(
                xi,
                xj,
                fs=fs,
                nperseg=nperseg,
                noverlap=noverlap,
                detrend="constant",
                scaling="density",
                return_onesided=True,
            )
            
            denom = (Sxx[i] * Sxx[j]) + EPS
            cxy = (np.abs(Sxy) ** 2) / denom

            for b, idx in enumerate(band_idxs):
                v = float(np.mean(cxy[idx])) if idx.size else 0.0
                coh[b, i, j] = v
                coh[b, j, i] = v # Symmetric

    return coh # Shape (Bands, Ch, Ch)