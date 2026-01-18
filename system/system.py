#system/system.py
import numpy as np
import os
from eeg.preprocessing import autopreprocess, segment_and_reference
from eeg.feature_extraction import compute_psd_epoch, compute_coh_epoch
from predictor.base import BaseModel

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

class EEGSystem:
    def __init__(self, models, fs=500, epoch_sec=2.0, debug=False):
        """
        models: dict {"EO":[model1,model2], "EC":[model3,...]}
        """
        self.models = models
        self.fs = fs
        self.epoch_sec = epoch_sec
        self.debug = debug

    def dbg(self, *args):
        if self.debug:
            print("[SYSTEM]", *args)

    ################################################
    # ORIGINAL: SINGLE FILE PIPELINE (KEEP)
    ################################################
    def run(
        self,
        csv_path,
        *,
        age,
        gender,
        education,
        sleep,
        well,
    ):
        self.dbg("=== PIPELINE START ===")

        ds = autopreprocess(csv_path, self.fs, print_debug=self.debug)

        segmented, err = segment_and_reference(
            ds, self.epoch_sec, self.fs, print_debug=self.debug
        )
        if err: return None, err

        n_epochs = segmented.shape[0]
        psd_list = []
        coh_list = []

        for i in range(n_epochs):
            ep = segmented[i]
            psd = compute_psd_epoch(ep, fs=self.fs)
            coh = compute_coh_epoch(ep, fs=self.fs, bands=COH_BANDS)

            if coh.ndim == 2:
                coh = coh[..., np.newaxis]
            elif coh.shape[0] <= 10:
                coh = np.transpose(coh, (1,2,0))

            # === DEBUG BLOCK ===
            self.dbg(f"[SINGLE] epoch {i}: psd={psd.shape} coh={coh.shape}")
            self.dbg("   psd sample:", psd.flatten()[:10])
            self.dbg("   coh stats: mean=%.4f min=%.4f max=%.4f" %
                    (coh.mean(), coh.min(), coh.max()))

            psd_list.append(psd)
            coh_list.append(coh)


        X_psd = np.array(psd_list)
        X_coh = np.array(coh_list)

        results = []

        for model in self.models:
            needed = model.inputs()

            if self.debug:
                self.dbg(f"[SYSTEM] >>> model: {model.name}")
                self.dbg(f"[SYSTEM]     needed inputs = {needed}")

            kwargs = {}
            if "psd" in needed: kwargs["psd"] = X_psd
            if "coh" in needed: kwargs["coh"] = X_coh
            if {"cont","cat"}.issubset(needed):
                if self.debug:
                    self.dbg("[SYSTEM]     demographic => forwarded")
                kwargs.update(dict(
                    age=age, gender=gender,
                    education=education, sleep=sleep, well=well
                ))

            if self.debug:
                self.dbg(f"[SYSTEM]     kwargs keys = {list(kwargs.keys())}")

            proba, classes = model.predict_proba(**kwargs)
            idx = model.hard_vote(proba)

            results.append({
                "model": model.name,
                "classes": classes,
                "epoch_probs": proba,
                "pred_idx": int(idx),
                "pred_label": classes[int(idx)]
            })

        return {"n_epochs": n_epochs, "results": results}, None
    


    ################################################
    # MULTI-FILE MULTI-CONDITION PIPELINE (PATCHED)
    ################################################
    def run_multi(
        self,
        cond_files,
        *,
        age,
        gender,
        education,
        sleep,
        well,
    ):
        results = []
        cache = {}

        for cond, flist in cond_files.items():
            if not flist:
                continue

            session_psd = []
            session_coh = []
            seg_last = None
            raw_last = None

            for f in flist:
                temp = f"__{f.name}"
                with open(temp, "wb") as g:
                    g.write(f.getbuffer())

                ds = autopreprocess(temp, self.fs, print_debug=self.debug)
                seg, err = segment_and_reference(ds, self.epoch_sec, self.fs, print_debug=self.debug)
                os.remove(temp)

                if err:
                    return None, err

                raw_last = ds
                seg_last = seg

                psd_list = []
                coh_list = []

            for i in range(seg.shape[0]):
                ep = seg[i]
                psd = compute_psd_epoch(ep, fs=self.fs)
                coh = compute_coh_epoch(ep, fs=self.fs, bands=COH_BANDS)

                if coh.ndim == 2:
                    coh = coh[..., np.newaxis]
                elif coh.shape[0] <= 10:
                    coh = np.transpose(coh, (1,2,0))

                # === DEBUG BLOCK ===
                self.dbg(f"[{cond}] epoch {i}: psd={psd.shape} coh={coh.shape}")
                self.dbg("   psd sample:", psd.flatten()[:10])
                self.dbg("   coh stats: mean=%.4f min=%.4f max=%.4f" %
                        (coh.mean(), coh.min(), coh.max()))

                psd_list.append(psd)
                coh_list.append(coh)

            session_psd.append(np.array(psd_list))
            session_coh.append(np.array(coh_list))

            cache[cond] = {
                "psd": np.concatenate(session_psd, axis=0),
                "coh": np.concatenate(session_coh, axis=0),
                "seg": seg_last,
                "raw": raw_last,
            }
            # after cache building
            X_coh = cache[cond]["coh"]
            if self.debug or True:   # force always print
                print("=== COHERENCE DEBUG ===")
                print(f"[COH] cond={cond} shape={X_coh.shape}")
                print(f"[COH] stats: mean={X_coh.mean():.4f} std={X_coh.std():.4f} min={X_coh.min():.4f} max={X_coh.max():.4f}")
                print("bands = [delta, theta, alpha, beta, gamma]")
                for bi, bn in enumerate(["delta","theta","alpha","beta","gamma"]):
                    print(f"[COH] band {bn} mean={X_coh[:,:,:,bi].mean():.4f}")
                print("=== END ===")


        for cond, models in self.models.items():
            if cond not in cache:
                continue

            X_psd = cache[cond]["psd"]
            X_coh = cache[cond]["coh"]

            if self.debug or True:   # force print
                print("=== PSD DEBUG ===")
                print(f"[PSD] cond={cond} shape={X_psd.shape}")
                print(f"[PSD] dtype={X_psd.dtype}")
                print(f"[PSD] stats: mean={X_psd.mean():.4f} std={X_psd.std():.4f} "
                    f"min={X_psd.min():.4f} max={X_psd.max():.4f}")

                # EPC: ch×freq
                E, C, F = X_psd.shape
                print(f"[PSD] epochs={E} channels={C} freqs={F}")

                # freq bands summary
                bands = [
                    ("delta", 1, 4),
                    ("theta", 4, 8),
                    ("alpha", 8, 13),
                    ("beta", 13, 30),
                    ("gamma", 30, 45),
                ]
                freqs = np.linspace(0, self.fs/2, F)

                for bn, lo, hi in bands:
                    idx = np.where((freqs>=lo)&(freqs<=hi))[0]
                    if len(idx)>0:
                        bmean = X_psd[:,:,idx].mean()
                        print(f"[PSD] band {bn:6s}: mean={bmean:.4f}")

                # sample few channel curves (mean over epochs)
                for ch in range(min(4,C)):
                    print(f"[PSD] ch{ch} first5:", X_psd[0,ch,:5], "...")


            for model in models:
                needed = model.inputs()
                if self.debug:
                    self.dbg(f"[MULTI] >>> model: {model.name} ({cond})")
                    self.dbg(f"[MULTI]     needed inputs = {needed}")
                
                kwargs = {}
                if "psd" in needed: kwargs["psd"] = X_psd
                if "coh" in needed: kwargs["coh"] = X_coh
                if {"cont","cat"}.issubset(needed):
                    if self.debug:
                        self.dbg("[MULTI]     demographic => forwarded")
                    kwargs.update(dict(
                        age=age, gender=gender,
                        education=education, sleep=sleep, well=well
                    ))

                if self.debug:
                    self.dbg(f"[MULTI]     kwargs keys = {list(kwargs.keys())}")

                proba, classes = model.predict_proba(**kwargs)
                idx = model.hard_vote(proba)

                results.append({
                    "model": f"{model.name} ({cond})",
                    "classes": classes,
                    "epoch_probs": proba,
                    "pred_idx": int(idx),
                    "pred_label": classes[int(idx)]
                })

        return {
            "results": results,
            "cache": cache,
        }, None
