# predictor/cnn_tab_EO_model.py

import os, json, joblib, numpy as np, tensorflow as tf
from keras import layers
from predictor.base import BaseModel

ARTIFACTS_DIR = "artifacts/CNNTab"

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
class CNNTabModel(BaseModel):
    def __init__(self, condition="EO", debug=False):
        assert condition in ["EO", "EC"]

        name = f"CNNTab_{condition}"
        super().__init__(name, n_classes=3)

        self.debug = debug
        self.condition = condition

        MODEL_FILE = (
            "tabtransformer_cnn_EO.best_new.keras" if condition=="EO"
            else "tabtransformer_cnn_EC.best.keras"
        )

        print(f"🔄 Loading CNN+TabTransformer ({condition})...")

        self.model = tf.keras.models.load_model(
            os.path.join(ARTIFACTS_DIR, MODEL_FILE),
            custom_objects={"CatSlice": CatSlice, "ColIndex": ColIndex},
            compile=False,
        )

        self.scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler_cont.pkl"))
        self.le = joblib.load(os.path.join(ARTIFACTS_DIR, "label_encoder.pkl"))
        self.cat_maps = json.load(open(os.path.join(ARTIFACTS_DIR, "cat_maps.json")))

        print(f"✅ {condition} model loaded")

    def inputs(self):
        return {"psd", "coh", "cont", "cat"}

    def encode_demo(self, age, gender, education, sleep, well, n_epochs):
        cont = self.scaler.transform([[age, education, sleep]])
        cont = np.repeat(cont, n_epochs, axis=0)

        g = self.cat_maps["gender"].get(str(gender), 0)
        w = self.cat_maps["well"].get(str(well), 0)
        cat = np.repeat(np.array([[g, w]], int), n_epochs, axis=0)

        return cont, cat

    def predict_proba(self, *, psd, coh, age, gender, education, sleep, well):
        n_epochs = psd.shape[0]
        cont, cat = self.encode_demo(age, gender, education, sleep, well, n_epochs)

        probs = self.model.predict({
            "psd": psd,
            "coh": coh,
            "cont": cont,
            "cat": cat
        }, verbose=0)

        return probs, self.le.classes_.tolist()
