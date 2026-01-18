# predictor/tab_model.py

import os, json, joblib, numpy as np, tensorflow as tf
from predictor.base import BaseModel
from keras import layers

ARTIFACTS_DIR = "artifacts/Tab"  

@tf.keras.utils.register_keras_serializable(package="Custom")
class CatSlice(layers.Layer):
    def __init__(self, idx, **kwargs):
        super().__init__(**kwargs)
        self.idx = idx

    def call(self, x):
        return tf.gather(x, indices=self.idx, axis=1)


@tf.keras.utils.register_keras_serializable(package="Custom")
class ColIndex(layers.Layer):
    def __init__(self, idx, **kwargs):
        super().__init__(**kwargs)
        self.idx = idx

    def call(self, x):
        b = tf.shape(x)[0]
        return tf.fill([b], tf.cast(self.idx, tf.int32))
    
class TabModel(BaseModel):
    def __init__(self, condition="EC", debug=False):
        assert condition in ["EO","EC"]
        name = f"TabTransformer_{condition}"
        super().__init__(name, n_classes=3)

        self.debug = debug
        self.condition = condition

        # đổi file theo phần training (best)
        MODEL_FILE = (
            "tabtransformer_EO.best.keras" if condition=="EO"
            else "tabtransformer_EC.best.keras"
        )

        print(f"🔄 Loading TabTransformer ({condition})...")

        # load keras
        self.model = tf.keras.models.load_model(
            os.path.join(ARTIFACTS_DIR, MODEL_FILE),
            custom_objects={"CatSlice": CatSlice, "ColIndex": ColIndex},
            compile=False
        )

        # load artifacts
        # === scaler ===
        scaler_cfg = json.load(open(os.path.join(ARTIFACTS_DIR,"cont_scaler.json")))
        self.scaler = scaler_cfg  # store

        # === labels ===
        lbl = json.load(open(os.path.join(ARTIFACTS_DIR,"labels.json")))
        self.labels = lbl["classes"]

        # === cat_maps ===
        self.cat_maps = json.load(open(os.path.join(ARTIFACTS_DIR,"cat_maps.json")))


        print(f"✅ {condition} TabTransformer loaded")

    def inputs(self):
        # không dùng coh
        return {"psd","cont","cat"}

    def encode_demo(self, age, gender, education, sleep, well, n_epochs):
    # === continuous ===
        try:
            x = np.array([[float(age), float(education), float(sleep)]], dtype=np.float32)
        except:
            raise ValueError(f"Invalid continuous input: age={age}, edu={education}, sleep={sleep}")

        mu = np.array(self.scaler["mean"], dtype=np.float32)
        sigma = np.array(self.scaler["scale"], dtype=np.float32)
        cont = (x - mu) / sigma
        cont = np.repeat(cont, n_epochs, axis=0)

        # === categorical ===
        g = self.cat_maps["gender"].get(str(gender), self.cat_maps["gender"].get("NA", 0))
        w = self.cat_maps["well"].get(str(well), self.cat_maps["well"].get("NA", 0))
        cat = np.repeat(np.array([[g,w]], dtype=np.int32), n_epochs, axis=0)

        return cont, cat

    def predict_proba(self, *, psd, age, gender, education, sleep, well):
        """
        psd: (n_epochs, 26,89,1)
        """
        n_epochs = psd.shape[0]
        cont, cat = self.encode_demo(age, gender, education, sleep, well, n_epochs)

        probs = self.model.predict({
            "psd": psd,
            "cont": cont,
            "cat": cat,
        }, verbose=0)

        return probs, self.labels

