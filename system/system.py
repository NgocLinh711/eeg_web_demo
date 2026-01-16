# system/system.py

import numpy as np
from eeg.preprocessing import autopreprocess, segment_and_reference
from eeg.feature_extraction import compute_psd_epoch, compute_coh_epoch
from predictor.base import BaseModel


class EEGSystem:
    def __init__(self, models: list[BaseModel], fs=500, epoch_sec=2.0, debug=False):
        """
        models: list các model (EO, EC, ensemble,...)
        """
        self.models = models
        self.fs = fs
        self.epoch_sec = epoch_sec
        self.debug = debug

    def dbg(self, *args):
        if self.debug:
            print("[SYSTEM]", *args)

    ################################################
    # MAIN PIPELINE
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

        # ------------------------------
        # STEP 1: Autopreprocess
        # ------------------------------
        ds = autopreprocess(csv_path, self.fs, print_debug=self.debug)

        # ------------------------------
        # STEP 2: Segment
        # ------------------------------
        segmented, err = segment_and_reference(
            ds, self.epoch_sec, self.fs, print_debug=self.debug
        )
        if err:
            return None, err

        n_epochs = segmented.shape[0]
        self.dbg(f"Epochs: {n_epochs}")

        # ------------------------------
        # STEP 3: Features (PSD + COH)
        # ------------------------------
        psd_list = []
        coh_list = []

        for i in range(n_epochs):
            ep = segmented[i]
            psd = compute_psd_epoch(ep, fs=self.fs)
            coh = compute_coh_epoch(ep, fs=self.fs)

            # standardize shape for CNN
            if coh.ndim == 2:
                coh = coh[..., np.newaxis]  # (ch, ch, 1)
            elif coh.shape[0] <= 10:        # assume (Bands, ch, ch)
                coh = np.transpose(coh, (1,2,0))  # → (ch, ch, Bands)

            psd_list.append(psd)
            coh_list.append(coh)

        X_psd = np.array(psd_list)
        X_coh = np.array(coh_list)

        # ------------------------------
        # STEP 4: Inference Multi-Model
        # ------------------------------
        results = []

        for model in self.models:
            needed = model.inputs()
            kwargs = {}

            if "psd" in needed: kwargs["psd"] = X_psd
            if "coh" in needed: kwargs["coh"] = X_coh

            if {"cont", "cat"}.issubset(needed):
                kwargs.update(dict(
                    age=age,
                    gender=gender,
                    education=education,
                    sleep=sleep,
                    well=well,
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

        self.dbg("=== PIPELINE DONE ===")
        return {
            "n_epochs": n_epochs,
            "results": results,
        }, None
