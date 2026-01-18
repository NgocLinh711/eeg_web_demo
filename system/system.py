import os
import time
import numpy as np

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

# Channel labels (26 electrodes TDBRAIN)
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
        models: dict e.g. {"EO":[model1,...], "EC":[model2,...]}
        """
        self.models = models
        self.fs = fs
        self.epoch_sec = epoch_sec
        self.debug = debug

    def dbg(self, *args):
        if self.debug: print("[SYSTEM]", *args)

    ##############################################################
    # ORIGINAL SINGLE-FILE API (unchanged, no timing)
    ##############################################################
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
        seg, err = segment_and_reference(ds, self.epoch_sec, self.fs, print_debug=self.debug)
        if err: return None, err

        n_epochs = seg.shape[0]
        psd_list, coh_list = [], []

        for i in range(n_epochs):
            ep = seg[i]
            psd = compute_psd_epoch(ep, fs=self.fs)
            coh = compute_coh_epoch(ep, fs=self.fs, bands=COH_BANDS)
            if coh.ndim == 2: coh = coh[..., np.newaxis]
            elif coh.shape[0] <= 10: coh = np.transpose(coh, (1,2,0))

            psd_list.append(psd)
            coh_list.append(coh)

        X_psd = np.array(psd_list)
        X_coh = np.array(coh_list)

        results = []
        for model in self.models:
            needed = model.inputs()
            kwargs = {}
            if "psd" in needed: kwargs["psd"] = X_psd
            if "coh" in needed: kwargs["coh"] = X_coh
            if {"cont","cat"}.issubset(needed):
                kwargs.update(dict(
                    age=age, gender=gender,
                    education=education, sleep=sleep, well=well
                ))

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


    ##############################################################
    # MULTI-FILE MULTI-CONDITION (EO / EC)
    ##############################################################
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
        t0_pipeline = time.perf_counter()

        # =============================
        # LOAD + CONCAT ALL FILES
        # =============================
        for cond, flist in cond_files.items():
            if not flist: continue

            print(f"\n=== MULTI-CONDITION START: {cond} ===")
            print(f"[{cond}] n_files = {len(flist)}")

            session_psd, session_coh = [], []
            seg_last, raw_last = None, None

            for idx, f in enumerate(flist):
                fname = f"__{f.name}"
                print(f"[{cond}] >>> loading {idx+1}/{len(flist)}: {f.name}")

                with open(fname,"wb") as g:
                    g.write(f.getbuffer())

                ds = autopreprocess(fname, self.fs, print_debug=self.debug)
                seg, err = segment_and_reference(ds, self.epoch_sec, self.fs, print_debug=self.debug)
                os.remove(fname)
                if err: return None, err

                seg_last, raw_last = seg, ds
                print(f"[{cond}] file {idx+1}: epochs = {seg.shape[0]}")

                psd_list, coh_list = [], []
                for i in range(seg.shape[0]):
                    ep = seg[i]
                    psd = compute_psd_epoch(ep, fs=self.fs)
                    coh = compute_coh_epoch(ep, fs=self.fs, bands=COH_BANDS)
                    if coh.ndim == 2: coh = coh[..., np.newaxis]
                    elif coh.shape[0] <= 10: coh = np.transpose(coh,(1,2,0))

                    psd_list.append(psd)
                    coh_list.append(coh)

                session_psd.append(np.array(psd_list))
                session_coh.append(np.array(coh_list))

            # concat all
            cache[cond] = {
                "psd": np.concatenate(session_psd, axis=0),
                "coh": np.concatenate(session_coh, axis=0),
                "seg": seg_last,
                "raw": raw_last,
            }
            print(f"[{cond}] TOTAL epochs = {cache[cond]['psd'].shape[0]}")

        # =============================
        # PREDICT FOR EACH CONDITION
        # =============================
        for cond, models in self.models.items():
            if cond not in cache: continue

            X_psd = cache[cond]["psd"]
            X_coh = cache[cond]["coh"]

            # --- PSD debug ---
            print(f"\n=== PSD DEBUG ({cond}) ===")
            print(f"[PSD] shape = {X_psd.shape}")
            print(f"[PSD] mean={X_psd.mean():.4f}, std={X_psd.std():.4f}, min={X_psd.min():.4f}, max={X_psd.max():.4f}")

            E,C,F = X_psd.shape
            print(f"[PSD] epochs={E}, channels={C}, freqs={F}")

            freqs = np.linspace(0, self.fs/2, F)
            for bn, lo, hi in COH_BANDS:
                idx = np.where((freqs>=lo)&(freqs<=hi))[0]
                if len(idx)>0:
                    bmean = X_psd[:,:,idx].mean()
                    print(f"[PSD] {bn:6s}: mean={bmean:.4f}")

            # preview channels
            print("--- PSD sample per channel ---")
            for ch in range(min(6,C)):
                print(f"  {CHANNEL_LABELS[ch]:>3s}: {X_psd[0,ch,:5]}")

            # =============================
            # TIMING PREDICT LOOP
            # =============================
            cond_total = 0.0

            for model in models:
                needed = model.inputs()
                kwargs = {}
                if "psd" in needed: kwargs["psd"] = X_psd
                if "coh" in needed: kwargs["coh"] = X_coh
                if {"cont","cat"}.issubset(needed):
                    kwargs.update(dict(
                        age=age, gender=gender,
                        education=education, sleep=sleep, well=well
                    ))

                t0 = time.perf_counter()
                proba, classes = model.predict_proba(**kwargs)
                t_pred = (time.perf_counter()-t0)*1000

                t1 = time.perf_counter()
                idx = model.hard_vote(proba)
                t_vote = (time.perf_counter()-t1)*1000

                t_total = t_pred + t_vote
                cond_total += t_total

                print(f"[{cond}] {model.name}: predict={t_pred:.1f}ms | vote={t_vote:.1f}ms | total={t_total:.1f}ms")

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

            print(f"[{cond}] TOTAL MODEL TIME = {cond_total:.1f}ms\n")

        pipeline_ms = (time.perf_counter()-t0_pipeline)*1000
        print(f"=== PIPELINE DONE in {pipeline_ms:.1f}ms ===")

        return {
            "results": results,
            "cache": cache,
        }, None
