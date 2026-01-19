# predictor/svm_model.py
import joblib
import numpy as np
import os
from predictor.base import BaseModel

class SVMModel(BaseModel):
    def __init__(self, condition="EC", debug=False):
        name = f"SVM_{condition}"
        super().__init__(name, n_classes=3)  # thay 3 bằng số lớp thực tế của bạn
        
        self.condition = condition
        self.debug = debug
        
        # Đường dẫn artifacts (thay đổi nếu cần)
        ARTIFACTS_DIR = f"artifacts/SVM" 
        
        self.model = joblib.load(os.path.join(ARTIFACTS_DIR, f"svm_{condition}_final.joblib"))
        self.scaler = joblib.load(os.path.join(ARTIFACTS_DIR, f"scaler.pkl"))
        self.le = joblib.load(os.path.join(ARTIFACTS_DIR, "label_encoder.pkl"))
        
        print(f"✅ Loaded SVM {condition} (PSD only)")

    def inputs(self):
        return {"psd"}  # Chỉ cần PSD, không cần coh, cont, cat

    def predict_proba(self, *, psd, coh=None, **kwargs):  # bỏ qua coh và tabular
        n_epochs = psd.shape[0]
        
        # Flatten PSD giống hệt lúc train: (n_epochs, n_ch * n_freq)
        X = psd.reshape(n_epochs, -1).astype(np.float32)
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Decision values → softmax approximate prob
        dec = self.model.decision_function(X_scaled)
        probs = self._softmax(dec)
        
        return probs, self.le.classes_.tolist()

    def _softmax(self, z):
        z = z - np.max(z, axis=1, keepdims=True)
        ez = np.exp(z)
        return ez / np.sum(ez, axis=1, keepdims=True)