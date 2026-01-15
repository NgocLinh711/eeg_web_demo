import os
import json
import joblib
import numpy as np
import tensorflow as tf
import keras
from keras import layers


# =========================
# Custom Layers (Required)
# =========================
# @keras.utils.register_keras_serializable(package="Custom")
# class CatSliceLayer(layers.Layer):
#     def __init__(self, idx, **kwargs):
#         super().__init__(**kwargs)
#         self.idx = idx

#     def call(self, x):
#         return x[:, self.idx]

#     def get_config(self):
#         config = super().get_config()
#         config.update({"idx": self.idx})
#         return config


# @keras.utils.register_keras_serializable(package="Custom")
# class ColIdxLayer(layers.Layer):
#     def __init__(self, idx, **kwargs):
#         super().__init__(**kwargs)
#         self.idx = idx

#     def call(self, x):
#         batch = tf.shape(x)[0]
#         return tf.fill([batch], self.idx)

#     def get_config(self):
#         config = super().get_config()
#         config.update({"idx": self.idx})
#         return config

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


# =========================
# Project utils (EEG RAW)
# =========================
from utils.autopreprocessing import dataset
from utils import interprocessing
from utils.feature_extraction import compute_psd_epoch, compute_coh_epoch


# =========================
# Config
# =========================
ARTIFACTS_DIR = "artifacts"
FS = 500
EPOCH_SEC = 2


# =========================
# Predictor System
# =========================
class PredictorSystem:
    def __init__(self):
        print("🔄 Loading model & artifacts...")

        model_path = os.path.join(
            ARTIFACTS_DIR,
            "tabtransformer_cnn_EO.best_new.keras"
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model not found: {model_path}")

        # Load model
        self.model = keras.models.load_model(
            model_path,
            custom_objects={
                # "CatSliceLayer": CatSliceLayer,
                # "ColIdxLayer": ColIdxLayer,
                "CatSlice": CatSlice,
                "ColIndex": ColIndex,
            },
            compile=False,
        )

        self.scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
        self.le = joblib.load(os.path.join(ARTIFACTS_DIR, "label_encoder.pkl"))

        with open(os.path.join(ARTIFACTS_DIR, "cat_maps.json"), "r", encoding="utf-8") as f:
            self.cat_maps = json.load(f)

        print("✅ Model loaded successfully!")

    # =========================
    # Predict Logic
    # =========================
    def process_and_predict(
        self,
        csv_path: str,
        age: float,
        gender,
        education: int,
        sleep: int,
        well,
    ):
        print("\n📂 STEP 1 — Load RAW EEG CSV")
        print(f" → File: {csv_path}")
        ds = dataset(csv_path, Fs=FS)
        print("[DEBUG] dataset created")
        print("[DEBUG] ds.Fs:", getattr(ds, "Fs", None), type(getattr(ds, "Fs", None)))
        print("[DEBUG] ds.info:", getattr(ds, "info", None))

        print("[DEBUG] -> loaddata()")
        ds.loaddata()
        print("[DEBUG] data shape:", ds.data.shape)
        print("[DEBUG] dtype:", ds.data.dtype)
        print("[DEBUG] loaddata OK")

        print("[DEBUG] -> bipolarEOG()")
        ds.bipolarEOG()
        print("[DEBUG] bipolarEOG OK")

        print("[DEBUG] -> apply_filters()")
        ds.apply_filters()
        print("[DEBUG] apply_filters OK")

        print("[DEBUG] -> correct_EOG()")
        ds.correct_EOG()
        print("[DEBUG] correct_EOG OK")

        print("[DEBUG] -> detect_emg()")
        ds.detect_emg()
        print("[DEBUG] detect_emg OK")

        print("[DEBUG] -> detect_jumps()")
        ds.detect_jumps()
        print("[DEBUG] detect_jumps OK")

        print("[DEBUG] -> detect_kurtosis()")
        ds.detect_kurtosis()
        print("[DEBUG] detect_kurtosis OK")

        print("[DEBUG] -> detect_extremevoltswing()")
        ds.detect_extremevoltswing()
        print("[DEBUG] detect_extremevoltswing OK")

        print("[DEBUG] -> residual_eyeblinks()")
        ds.residual_eyeblinks()
        print("[DEBUG] residual_eyeblinks OK")

        print("[DEBUG] -> define_artifacts()")
        ds.define_artifacts()
        print("[DEBUG] define_artifacts OK")


        raw_data = ds.data[:26, :]   # 26 EEG channel
        n_channels, n_samples = raw_data.shape
        duration_sec = n_samples / FS

        print(f"   RAW shape: {raw_data.shape}  (26 ch x {n_samples} samples)")
        print(f"   Duration: {duration_sec:.2f} sec")

        # ─────────────────────────────────────────
        # STEP 1.1 — AUTO-PREPROCESSING TRACKING
        # ─────────────────────────────────────────
        if hasattr(ds, "info"):
            print("\n🧼 AUTO-PREPROCESSING INFO")
            for k,v in ds.info.items():
                print(f"   - {k}: {v}")

        if hasattr(ds, "artifacts"):
            for k,v in ds.artifacts.items():
                print(f"   - {k}: {v}")

        if duration_sec < EPOCH_SEC:
            print("❌ EEG too short pre-segmentation")
            return None, "EEG too short"


        print("\n🧩 STEP 2 — Build interdataset & Segment Epoch 2s")

        input_3d = raw_data[np.newaxis, :, :]
        inter = interprocessing.interdataset({
            "data": input_3d,
            "labels": [f"ch{i}" for i in range(30)],
            "Fs": FS,
            "info": {"fileID": csv_path},
        })

        # # ==== ADD ARTIFACT CHANNEL IF EXISTS ====
        # if hasattr(ds, "artifact_mask"):
        #     artifact = ds.artifact_mask[np.newaxis, np.newaxis, :]  # (1,1,T)
        #     inter.data = np.concatenate([inter.data, artifact], axis=1)
        #     inter.labels.append("artifacts")

        print("   Before segment:")
        print(f"     → data: {inter.data.shape}   (1 x 26 x {n_samples})")
        print(f"     → duration: {duration_sec:.2f} sec")

        print(f" → Calling segment(marking='no', trllength={EPOCH_SEC}, remove_artifact='yes')")
        inter.segment(marking='no', trllength=EPOCH_SEC, remove_artifact='yes')

        segmented = inter.data
        n_epochs = segmented.shape[0]
        epoch_len = segmented.shape[-1] / FS if n_epochs > 0 else 0

        print("   After segment:")
        print(f"     → segmented: {segmented.shape}    ({n_epochs} epochs, each {epoch_len:.2f} sec)")

        if hasattr(inter, "arttrl"):
            print(f"     → artifact segments: {inter.arttrl}")

        if hasattr(inter, "info"):
            print("   Segment info:")
            for k,v in inter.info.items():
                print(f"     - {k}: {v}")

        if n_epochs == 0:
            print("❌ Not enough valid segments after artifact removal")
            return None, "EEG too short or all removed by artifacts"

        print("[CHECK] labels:", inter.labels[:30])
        print("[CHECK] artifacts:", inter.info.get('artifact removal'))
        print("[CHECK] quality:", inter.info.get('data quality'))
        print("[CHECK] num epochs:", n_epochs)
        print("[CHECK] artifact removal:", inter.info.get('artifact removal', None))
        print("[CHECK] data quality:", inter.info.get('data quality', None))


        print("\n🧮 STEP 3 — Compute PSD + COH features")
        psd_list, coh_list = [], []
        for i, ep in enumerate(segmented):
            psd = compute_psd_epoch(ep, fs=FS)
            coh = compute_coh_epoch(ep, fs=FS)
            psd_list.append(psd)
            coh_list.append(coh)
            print(f"   → Epoch {i+1}/{n_epochs}: PSD={psd.shape}, COH={coh.shape}")

        X_psd = np.asarray(psd_list, dtype=np.float32)[..., np.newaxis]
        X_coh = np.asarray(coh_list, dtype=np.float32).transpose(0, 2, 3, 1)
        print(f"   Final PSD shape: {X_psd.shape}   # (E,26,89,1)")
        print(f"   Final COH shape: {X_coh.shape}   # (E,26,26,5)")

        print("\n🔢 STEP 4 — Build CONT + CAT features")
        cont_vec = self.scaler.transform(
            np.array([[age, education, sleep]], dtype=np.float32)
        )
        X_cont = np.repeat(cont_vec, n_epochs, axis=0)
        print(f"   CONT shape: {X_cont.shape}  → {cont_vec}")

        g_val = self.cat_maps["gender"].get(str(gender), 0)
        w_val = self.cat_maps["well"].get(str(well), 0)
        X_cat = np.repeat(np.array([[g_val, w_val]], dtype=np.int32), n_epochs, axis=0)
        print(f"   CAT shape: {X_cat.shape}   → gender={g_val}, well={w_val}")

        print("\n🤖 STEP 5 — Model Inference")
        probs = self.model.predict(
            {"psd": X_psd, "coh": X_coh, "cont": X_cont, "cat": X_cat},
            verbose=0,
        )
        print(f"   Output probs shape: {probs.shape}")

        classes = self.le.classes_
        print(f"   Classes: {classes.tolist()}")

        print("\n✅ DONE — Returning prediction\n")

        return probs, classes
