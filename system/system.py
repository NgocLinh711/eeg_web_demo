# =============================
# system/system.py (Cloud-Safe v2)
# =============================

import os, time, warnings
import numpy as np
warnings.filterwarnings("ignore")

from eeg.preprocessing import autopreprocess, segment_and_reference
from eeg.feature_extraction import compute_psd_epoch, compute_coh_epoch

# Limit thread pools (RF/XGB/NumPy/BLAS)
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
os.environ.setdefault("NUMEXPR_NUM_THREADS","1")

FS = 500.0
COH_BANDS = [
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 45.0),
]

CHANNEL_LABELS = [
    "Fp1","Fp2","F7","F3","Fz","F4","F8",
    "FC5","FC1","FC2","FC6",
    "T7","C3","Cz","C4","T8",
    "TP9","CP5","CP1","CP2","CP6","TP10",
    "P7","P3","Pz","P4","P8","O1","Oz","O2"
][:26]


class EEGSystem:

    def __init__(self, models, fs=500, epoch_sec=2.0, debug=False):
        """
        models = {"EO":[...], "EC":[...]}
        """
        self.models = models
        self.fs = fs
        self.epoch_sec = epoch_sec
        self.debug = debug

    def dbg(self, *msg):
        if self.debug: print("[SYSTEM]", *msg)

    # ===========================================
    # MULTI-CONDITION PIPELINE
    # ===========================================
    def run_multi(
        self,
        cond_files,
        *,
        age,
        gender,
        education,
        sleep,
        well,
        debug_coh_per_file=False
    ):
        results, cache = [], {}
        t0_pipeline = time.perf_counter()

        # -----------------------------
        # 1) LOAD + FEATURE EXTRACTION
        # -----------------------------
        t0_pre = time.perf_counter()     # <--- bắt đầu tiền xử lý

        for cond, flist in cond_files.items():
            if not flist: continue
            self.dbg(f"\n=== CONDITION {cond} ==")

            session_psd, session_coh = [], []
            seg_last, raw_last = None, None

            for idx, f in enumerate(flist):
                tmp = f"__{f.name}"
                with open(tmp, "wb") as g:
                    g.write(f.getbuffer())

                ds = autopreprocess(tmp, self.fs, print_debug=self.debug)
                seg, err = segment_and_reference(ds, self.epoch_sec, self.fs, print_debug=self.debug)
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except PermissionError:
                        pass

                if err: return None, err
                seg_last, raw_last = seg, ds

                psd_list, coh_list = [], []
                for ep in seg:
                    psd = compute_psd_epoch(ep, fs=self.fs)
                    coh = compute_coh_epoch(ep, fs=self.fs, bands=COH_BANDS)
                    if coh.ndim == 2: coh = coh[...,None]
                    elif coh.shape[0] <= 10: coh = coh.transpose(1,2,0)

                    psd_list.append(psd)
                    coh_list.append(coh)

                session_psd.append(np.array(psd_list))
                session_coh.append(np.array(coh_list))

            cache[cond] = {
                "psd": np.concatenate(session_psd),
                "coh": np.concatenate(session_coh),
                "seg": seg_last,
                "raw": raw_last
            }
        t_pre = (time.perf_counter() - t0_pre) * 1000     # <--- ms
        self.dbg(f"[PREPROCESS] {round(t_pre,1)} ms")

        # -----------------------------
        # 2) PREDICT (CLOUD SAFE)
        # -----------------------------
        import tensorflow as tf
        tf.keras.backend.clear_session()

        for cond, models in self.models.items():
            if cond not in cache: continue

            X_psd = cache[cond]["psd"]
            X_coh = cache[cond]["coh"]

            # --- DEBUG section (only once per condition)
            if self.debug:
                print(f"\n=== DEBUG ({cond}) ===")

                # COH
                print(f"[COH] shape={X_coh.shape}")
                print(f"[COH] stats: mean={X_coh.mean():.4f}, std={X_coh.std():.4f}, "
                    f"min={X_coh.min():.4f}, max={X_coh.max():.4f}")

                band_names = ["delta","theta","alpha","beta","gamma"]
                for bi, bn in enumerate(band_names):
                    b = X_coh[:,:,:,bi]
                    print(f"[COH] {bn:6s}: mean={b.mean():.4f}, min={b.min():.4f}, max={b.max():.4f}")

                # preview (optional)
                bidx = 2  # alpha
                b = X_coh[:,:,:,bidx]
                print(f"\n[COH] preview (band={band_names[bidx]}):")
                pairs = [(0,1), (0,3)]  # Fp1-Fp2, Fp1-F3
                for (i,j) in pairs:
                    arr = b[:, i, j]
                    print(f"    {CHANNEL_LABELS[i]}-{CHANNEL_LABELS[j]}: {np.round(arr[:5],3)}")

                # === PSD ===
                print(f"\n[PSD] shape={X_psd.shape}")
                print(f"[PSD] stats: mean={X_psd.mean():.4f}, std={X_psd.std():.4f}, "
                    f"min={X_psd.min():.4f}, max={X_psd.max():.4f}")

                E,C,F = X_psd.shape
                freqs = np.linspace(0, self.fs/2, F)

                for bn, lo, hi in COH_BANDS:
                    idx = np.where((freqs>=lo)&(freqs<=hi))[0]
                    if len(idx)>0:
                        b = X_psd[:,:,idx]
                        print(f"[PSD] {bn:6s}: mean={b.mean():.4f}, min={b.min():.4f}, max={b.max():.4f}")

                # preview channel
                ch = 0
                print(f"\n[PSD] preview channel: {CHANNEL_LABELS[ch]}")
                print(f"    freqs→ {np.round(freqs[:5],1)}")
                print(f"    values→ {np.round(X_psd[0,ch,:5],3)}")

            E = X_psd.shape[0]
            self.dbg(f"\n=== PREDICT {cond} (epochs={E}) ===")

            cond_time = 0.0

            def safe_predict(model, **kwargs):
                try:
                    proba, classes = model.predict_proba(**kwargs)
                    return proba, classes, None
                except Exception as e:
                    return None, None, str(e)

            for model in models:
                needed = model.inputs()
                kwargs = {}

                if "psd" in needed: kwargs["psd"] = X_psd
                if "coh" in needed: kwargs["coh"] = X_coh
                if {"cont","cat"}.issubset(needed) or "cat" in needed:
                    kwargs.update(dict(
                        age=age, gender=gender,
                        education=education,
                        sleep=sleep, well=well,
                    ))

                t0 = time.perf_counter()
                proba, classes, err = safe_predict(model, **kwargs)
                t_pred = (time.perf_counter()-t0)*1000

                if err:
                    results.append({
                        "model": f"{model.name} ({cond})",
                        "error": err
                    })
                    continue

                t1 = time.perf_counter()
                idx = model.hard_vote(proba)
                t_vote = (time.perf_counter()-t1)*1000
                t_total = t_pred + t_vote
                cond_time += t_total

                results.append({
                    "model": f"{model.name} ({cond})",
                    "classes": classes,
                    "epoch_probs": proba,
                    "pred_idx": int(idx),
                    "pred_label": classes[int(idx)],
                    "t_predict_ms": round(t_pred,2),
                    "t_vote_ms": round(t_vote,2),
                    "t_total_ms": round(t_total,2),
                })

                print(f"[PREDICT] {model.name} ({cond}): pred={t_pred:.1f} ms vote={t_vote:.1f} ms total={t_total:.1f} ms")

                tf.keras.backend.clear_session()

            self.dbg(f"[{cond}] total={round(cond_time,1)} ms")

        pipeline_ms = (time.perf_counter()-t0_pipeline)*1000
        self.dbg(f"\n=== DONE {round(pipeline_ms,1)} ms ===")

        return {
            "results": results,
            "cache": cache,
            "preprocess_ms": round(t_pre,1),   # <---
            "pipeline_ms": round(pipeline_ms,1),
        }, None
 