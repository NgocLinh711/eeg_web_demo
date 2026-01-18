import os, json, joblib
import numpy as np
from predictor.base import BaseModel

ARTIFACTS_DIR = "artifacts/RF"

class RFModel(BaseModel):
    def __init__(self, condition="EC", debug=False):
        assert condition in ["EO","EC"]
        self.condition = condition
        self.debug = debug
        name = f"RF({condition})"
        super().__init__(name, n_classes=3)

        # load model
        model_path = os.path.join(ARTIFACTS_DIR, f"rf_{condition}.pkl")
        self.model = joblib.load(model_path)

        # load label encoder
        self.le = joblib.load(os.path.join(ARTIFACTS_DIR, "label_encoder.pkl"))

        # load feature config
        with open(os.path.join(ARTIFACTS_DIR, "feature_config.json")) as f:
            cfg = json.load(f)

        self.classes = cfg["classes"]
        self.feature_dim = cfg["feature_dim"]
        self.use_psd = cfg["USE_PSD"]

    def inputs(self):
        return {"psd"}

    def _flat(self, psd):
        return psd.reshape(psd.shape[0], -1).astype(np.float32)

    def predict_proba(self, *, psd=None, **_):
        if psd is None:
            raise ValueError("RFModel requires PSD input")

        X = self._flat(psd)

        if X.shape[1] != self.feature_dim:
            raise ValueError(f"feature mismatch: {X.shape[1]} vs {self.feature_dim}")

        prob = self.model.predict_proba(X)
        return prob, self.classes
