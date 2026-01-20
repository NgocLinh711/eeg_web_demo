# predictor/cnn_model.py

import os, json, numpy as np, tensorflow as tf
from predictor.base import BaseModel

ARTIFACTS_DIR = "artifacts/CNN"

class CNNModel(BaseModel):
    def __init__(self, condition="EC", debug=False):
        assert condition in ["EO","EC"]
        name = f"CNN_{condition}"
        super().__init__(name, n_classes=3)

        self.condition = condition
        self.debug = debug

        MODEL_FILE = (
            "cnn_EO.best.keras" if condition=="EO"
            else "cnn_EC.best.keras"
        )

        print(f"🔄 Loading CNN-Coherence ({condition})...")

        self.model = tf.keras.models.load_model(
            os.path.join(ARTIFACTS_DIR, MODEL_FILE),
            compile=False
        )

        # minimal artifacts
        lbl = json.load(open(os.path.join(ARTIFACTS_DIR,"labels.json")))
        self.labels = lbl["classes"]

        self.config   = json.load(open(os.path.join(ARTIFACTS_DIR,"config.json")))
        self.pipeline = json.load(open(os.path.join(ARTIFACTS_DIR,"pipeline.json")))

        print(f"✅ {condition} CNN-Coherence loaded")

    def inputs(self):
        return {"coh"}

    def predict_proba(self, *, coh, **kwargs):
        if self.debug:
            print("[CNN-COH] coh:", coh.shape)
        probs = self.model.predict({"coh": coh}, verbose=0)
        return probs, self.labels
