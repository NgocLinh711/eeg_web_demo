import os
import json
import joblib
import numpy as np
import tensorflow as tf
import keras
from keras import layers

################################################
# Custom Layers for Keras Load
################################################
@tf.keras.utils.register_keras_serializable(package="Custom")
class CatSlice(layers.Layer):
    def __init__(self, idx, **kwargs):
        super().__init__(**kwargs)
        self.idx = int(idx)
    def call(self, inputs):
        return tf.gather(inputs, indices=self.idx, axis=1)
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"idx": self.idx})
        return cfg


@tf.keras.utils.register_keras_serializable(package="Custom")
class ColIndex(layers.Layer):
    def __init__(self, idx, **kwargs):
        super().__init__(**kwargs)
        self.idx = int(idx)
    def call(self, inputs):
        b = tf.shape(inputs)[0]
        return tf.fill([b], tf.cast(self.idx, tf.int32))
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"idx": self.idx})
        return cfg


################################################
# EEG Project Utils
################################################
# Assumes autopreprocessing.py is in the same folder or properly installed
try:
    from utils.autopreprocessing import dataset
    from utils.feature_extraction import compute_psd_epoch, compute_coh_epoch
    from utils.interprocessing import interdataset
except ImportError:
    # Fallback if files are in the root directory
    from autopreprocessing import dataset
    # You must have feature_extraction.py available
    from feature_extraction import compute_psd_epoch, compute_coh_epoch
    from interprocessing import interdataset


################################################
# Config
################################################
ARTIFACTS_DIR = "artifacts"
FS = 500
EPOCH_SEC = 2


################################################
# Predictor
################################################
class PredictorSystem:
    def __init__(self, debug=True):
        self.debug = debug

        print("🔄 Loading model & artifacts...")

        model_path = os.path.join(
            ARTIFACTS_DIR,
            "tabtransformer_cnn_EO.best_new.keras"
        )

        self.model = keras.models.load_model(
            model_path,
            custom_objects={
                "CatSlice": CatSlice,
                "ColIndex": ColIndex,
            },
            compile=False,
        )

        self.scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler_cont.pkl"))
        self.le = joblib.load(os.path.join(ARTIFACTS_DIR, "label_encoder.pkl"))

        with open(os.path.join(ARTIFACTS_DIR, "cat_maps.json"), "r", encoding="utf-8") as f:
            self.cat_maps = json.load(f)

        print("✅ Model loaded successfully!")

    ################################################
    # Debug helper
    ################################################
    def dbg(self, *args):
        if self.debug:
            print("[DEBUG]", *args)

    ################################################
    # Main Predict
    ################################################
    def process_and_predict(
        self,
        csv_path: str,
        age: float,
        gender,
        education: int,
        sleep: int,
        well,
    ):
        ################################################
        # STEP 1 — Load RAW + Autoprocess
        ################################################
        print("\n📂 STEP 1 — Load RAW EEG + Autoprocess")
        self.dbg("CSV:", csv_path)

        ds = dataset(csv_path, Fs=FS)
        ds.loaddata()
        self.dbg("Raw loaded:", ds.data.shape, ds.data.dtype)

        # Run pipeline
        ds.bipolarEOG()
        ds.apply_filters()
        ds.correct_EOG()
        ds.detect_emg()
        ds.detect_jumps()
        ds.detect_kurtosis()
        ds.detect_extremevoltswing()
        ds.residual_eyeblinks()
        ds.define_artifacts()

        raw_data = ds.data[:26,:]
        dur = raw_data.shape[1] / FS
        self.dbg("Duration:", dur, "sec")

        if dur < EPOCH_SEC:
            return None, "EEG too short"

        ################################################
        # ARTIFACT REPORT
        ################################################
        print("\n🧼 AUTO-PREPROCESS INFO")
        for k,v in ds.info.items():
            self.dbg(f"{k}: {v}")

        # Make data 3D for interdataset compatibility
        ds.data = np.expand_dims(ds.data, axis=0)

        # Wrap into interdataset for additional methods
        ds = interdataset(ds.__dict__)

        ################################################
        # STEP 2 — Rereference
        ################################################
        print("\n🧩 STEP 2 — AvgRef")
        ds.rereference('avgref')

        ################################################
        # STEP 3 — Segment
        ################################################
        print("\n🧩 STEP 3 — Segment 2s remove_artifact=yes")

        try:
            ds.segment(
                trllength=float(EPOCH_SEC),
                remove_artifact="yes",
                marking="no",
            )
        except Exception as e:
            self.dbg("SEGMENT FAILED:", e)
            return None, "segment-failed"

        # Keep only the first 26 EEG channels
        segmented = ds.data[:, :26, :]
        labels = ds.labels[:26]

        self.dbg("Segmented:", segmented.shape)
        
        n_epochs, n_ch, n_samp = segmented.shape
        self.dbg("Epochs:", n_epochs, "Channels:", n_ch)

        if n_epochs == 0:
            return None, "no-epochs"
        if n_ch != 26:
            return None, f"unexpected EEG channels: {n_ch}, expect 26"

        ################################################
        # STEP 4 — PSD + COH
        ################################################
        print("\n🧮 STEP 4 — PSD & COH")
        psd_list, coh_list = [], []

        for i in range(n_epochs):
            ep = segmented[i]

            psd = compute_psd_epoch(ep, fs=FS)     # (26, F)
            coh = compute_coh_epoch(ep, fs=FS)     # (26,26) OR (5,26,26)

            # normalize shape for model
            if coh.ndim == 2:
                coh = coh[..., np.newaxis]        # → (26,26,1)
            elif coh.shape[0] == 5 and coh.ndim == 3:
                coh = np.transpose(coh, (1,2,0))  # (5,26,26) → (26,26,5)

            psd_list.append(psd)
            coh_list.append(coh)
        
        X_psd = np.array(psd_list) # Shape: (n_epochs, 26, F)
        X_coh = np.array(coh_list) # Shape: (n_epochs, 26, 26, 5)
        self.dbg("X_psd shape:", X_psd.shape)
        self.dbg("X_coh shape:", X_coh.shape)

        ################################################
        # STEP 5 — CONT + CAT
        ################################################
        print("\n🔢 STEP 5 — CONT & CAT")

        cont_vec = self.scaler.transform(
            np.array([[age, education, sleep]], dtype=np.float32)
        )
        X_cont = np.repeat(cont_vec, n_epochs, axis=0)

        g_val = self.cat_maps["gender"].get(str(gender), 0)
        w_val = self.cat_maps["well"].get(str(well), 0)
        X_cat = np.repeat(np.array([[g_val, w_val]], dtype=np.int32), n_epochs, axis=0)

        ################################################
        # STEP 6 — Inference
        ################################################
        print("\n🤖 STEP 6 — Inference Model")

        probs = self.model.predict(
            {"psd": X_psd, "coh": X_coh, "cont": X_cont, "cat": X_cat},
            verbose=0,
        )

        classes = self.le.classes_
        p_mean = probs.mean(axis=0)
        idx = np.argmax(p_mean)
        pred = classes[idx]

        print("\n🎯 RESULT")
        self.dbg("Classes:", classes.tolist())
        self.dbg("Mean probs:", p_mean)
        print(f" → pred: {pred}")
        print(f" → n_epochs={n_epochs}")

        return {
            "pred_label": pred,
            "mean_prob": p_mean,
            "epoch_probs": probs,
            "classes": classes.tolist(),
            "n_epochs": n_epochs,
        }, None