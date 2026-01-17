import os, json, joblib, numpy as np, tensorflow as tf
from keras import layers
from predictor.base import BaseModel

ARTIFACTS_DIR = "artifacts/MLP"
MODEL_FILE = "mlp_EC.best.keras"

@tf.keras.utils.register_keras_serializable(package="Custom")
class CatSlice(layers.Layer):
    def __init__(self, idx, **kwargs):
        super().__init__(**kwargs)
        self.idx = idx

    def call(self, x):
        return tf.gather(x, indices=self.idx, axis=1)

@tf.keras.utils.register_keras_serializable(package="Custom")
class EmptyFeatures(layers.Layer):
    """Tạo tensor rỗng (B, 0) khi không có đặc trưng categorical nào"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        # Dùng inputs chỉ để lấy batch size
        return tf.zeros((tf.shape(inputs)[0], 0), dtype=tf.float32)

    def get_config(self):
        return super().get_config()


class MLPModel(BaseModel):
    def __init__(self, debug=False):
        assert condition in ["EO", "EC"]

        name = f"MLP_{condition}"
        super().__init__(name, n_classes=3)

        self.debug = debug
        self.condition = condition

        print("🔄 Loading MLP (EC)...")

        self.model = tf.keras.models.load_model(
            os.path.join(ARTIFACTS_DIR, MODEL_FILE),
            custom_objects={"CatSlice": CatSlice, "EmptyFeatures": EmptyFeatures},
            compile=False,
        )

        self.scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler_cont.pkl"))
        self.le = joblib.load(os.path.join(ARTIFACTS_DIR, "label_encoder.pkl"))
        self.cat_maps = json.load(open(os.path.join(ARTIFACTS_DIR, "cat_maps.json")))

        print("✅ EC model loaded")

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