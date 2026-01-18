# predictor/xgb_model.py

import os
import json
import joblib
import numpy as np
from xgboost import XGBClassifier
from predictor.base import BaseModel

ARTIFACTS_DIR = "artifacts/XGBoost"  

class XGBModel(BaseModel):
    def __init__(self, condition="EO", debug=False):
        assert condition in ["EO", "EC"]

        self.condition = condition
        self.debug = debug

        # Model name style consistent with CNN-Tab / CNN / Tab
        name = f"XGB_{condition}"
        super().__init__(name, n_classes=3)

        # ================================
        # Resolve artifact directory
        # ================================
        self.artifact_dir = os.path.join(ARTIFACTS_DIR)

        # Resolve model filename just like CNNTab logic
        MODEL_FILE = (
            f"xgb_{condition.lower()}.json"
        )

        model_path = os.path.join(self.artifact_dir, MODEL_FILE)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing: {model_path}")

        print(f"🔄 Loading XGBoost ({condition})...")

        # ================================
        # Load model
        # ================================
        self.model = XGBClassifier()
        self.model.load_model(model_path)

        # ================================
        # Load encoder + config
        # ================================
        self.le = joblib.load(os.path.join(self.artifact_dir, "label_encoder.pkl"))

        with open(os.path.join(self.artifact_dir, "feature_config.json"), "r") as f:
            self.config = json.load(f)

        with open(os.path.join(self.artifact_dir, "meta.json"), "r") as f:
            self.meta = json.load(f)

        self.classes = self.config["classes"]
        self.feature_dim = self.meta["feature_dim"]
        self.use_psd = self.config["USE_PSD"]

        if self.debug:
            print(f"[XGB-{condition}] classes={self.classes}, feat_dim={self.feature_dim}")

    def inputs(self):
        return {"psd"}

    def _flatten_psd(self, arr):
        # arr: (epochs, ch, freq)
        return arr.reshape(arr.shape[0], -1).astype(np.float32)

    def predict_proba(self, *, psd=None, **kwargs):
        if psd is None:
            raise ValueError("XGBModel requires 'psd' input")

        X = self._flatten_psd(psd)

        if X.shape[1] != self.feature_dim:
            raise ValueError(
                f"[XGB-{self.condition}] feature mismatch: expected={self.feature_dim}, got={X.shape[1]}"
            )

        prob = self.model.predict_proba(X)
        return prob, self.classes
