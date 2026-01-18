# predictor/rf_model.py
import os
import json
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from predictor.base import BaseModel

class RFModel(BaseModel):
    """
    RandomForest inference for EEG
    Compatible w/ EEGSystem (system/system.py)
    """

    def __init__(self, condition="EC", artifact_root="web_demo/model", debug=False):
        assert condition in ["EO", "EC"]

        self.condition = condition
        self.debug = debug

        # model naming convention like CNNTab + XGB
        name = f"RF_{condition}"
        super().__init__(name, n_classes=3)

        self.artifact_dir = artifact_root

        MODEL_FILE = f"rf_{condition.lower()}.pkl"
        model_path = os.path.join(self.artifact_dir, MODEL_FILE)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"RFModel missing: {model_path}")

        print(f"🔄 Loading RandomForest ({condition}) from: {model_path}")
        self.model = joblib.load(model_path)

        # load encoder (shared for both EO/EC)
        le_path = os.path.join(self.artifact_dir, "label_encoder.pkl")
        self.le = joblib.load(le_path)

        # config
        with open(os.path.join(self.artifact_dir, "feature_config.json"), "r") as f:
            self.config = json.load(f)

        with open(os.path.join(self.artifact_dir, "meta.json"), "r") as f:
            self.meta = json.load(f)

        self.classes = self.config["classes"]
        self.feature_dim = self.meta["feature_dim"]
        self.use_psd = self.config["USE_PSD"]

        if self.debug:
            print(f"[RF-{condition}] feat_dim={self.feature_dim}, classes={self.classes}")

    # RF chỉ cần PSD, không demog + không coherence
    def inputs(self):
        return {"psd"}

    def _flatten_psd(self, arr):
        # arr: (epochs, ch, freq)
        return arr.reshape(arr.shape[0], -1).astype(np.float32)

    def predict_proba(self, *, psd=None, **kwargs):
        if psd is None:
            raise ValueError("RFModel requires 'psd' input")

        X = self._flatten_psd(psd)

        if X.shape[1] != self.feature_dim:
            raise ValueError(
                f"[RF-{self.condition}] mismatch: expected={self.feature_dim}, got={X.shape[1]}"
            )

        prob = self.model.predict_proba(X)
        return prob, self.classes
