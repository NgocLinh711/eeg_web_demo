import streamlit as st
import numpy as np
import time
import os
import pandas as pd

from system.system import EEGSystem

# ============ PREDICTOR IMPORTS =============
from predictor.cnn_tab_EO_model import CNNTabEOModel
# from predictor.cnn_tab_EC_model import CNNTabECModel

# from predictor.cnn_EO_model import CNNEOModel
# from predictor.cnn_EC_model import CNNECModel

# from predictor.tab_EO_model import TabEOModel
# from predictor.tab_EC_model import TabECModel

# from predictor.mlp_EO_model import MLPEOModel
# from predictor.mlp_EC_model import MLPECModel

# from predictor.svm_EO_model import SVMEOMLModel
# from predictor.svm_EC_model import SVMECModel

# from predictor.xgb_EO_model import XGBEOModel
# from predictor.xgb_EC_model import XGBECModel

# from predictor.rf_EO_model import RFEOModel
# from predictor.rf_EC_model import RFECModel


###############################################
# MODEL REGISTRY (by type + condition)
###############################################
@st.cache_resource
def load_registry():

    registry = {
        "CNN-Tab": {
            "EO": CNNTabEOModel(debug=False),
            # "EC": CNNTabECModel(debug=False),
        },
        "CNN": {
            # "EO": CNNEOModel(debug=False),
            # "EC": CNNECModel(debug=False),
        },
        "Tab": {
            # "EO": TabEOModel(),
            # "EC": TabECModel(),
        },
        "MLP": {
            # "EO": MLPEOModel(),
            # "EC": MLPECModel(),
        },
        "SVM": {
            # "EO": SVMEOMLModel(),
            # "EC": SVMECLModel(),
        },
        "XGBoost": {
            # "EO": XGBEOModel(),
            # "EC": XGBECModel(),
        },
        "Random Forest": {
            # "EO": RFEOModel(),
            # "EC": RFECModel(),
        }
    }

    return registry


###############################################
# STREAMLIT UI
###############################################
st.set_page_config(page_title="EEG Diagnostic System", layout="wide", page_icon="🧠")

with st.sidebar:
    st.header("Thông tin bệnh nhân")

    age = st.number_input("Tuổi", 6.0, 100.0, 25.0)
    gender = st.selectbox("Giới tính", [1, 0], format_func=lambda x: "Nam" if x==1 else "Nữ")
    education = st.number_input("Học vấn (năm)", 0.0, 30.0, 12.0)
    sleep = st.number_input("Giờ ngủ", 0.0, 24.0, 7.0)
    well = st.selectbox("Well-being", [-2,-1,0,1,2,3], index=2)

    st.markdown("---")

    st.subheader("Chọn mô hình")
    model_types = st.multiselect(
        "Loại mô hình:",
        ["CNN-Tab", "CNN", "Tab", "MLP", "SVM", "XGBoost", "Random Forest"],
        default=["CNN-Tab"]
    )

    st.subheader("Condition")
    condition = st.selectbox("Điều kiện EEG", ["EO", "EC", "Both"])

st.title("🧠 EEG Multi-Model Diagnostic System")
st.markdown("TabTransformer • CNN • Classical ML • EO/EC Conditions")
st.markdown("---")


file = st.file_uploader("📤 Upload EEG Raw CSV", type=["csv"])


###############################################
# DIAGNOSIS RUN
###############################################
if file and st.button("🚀 Chạy chẩn đoán", type="primary"):

    registry = load_registry()

    # Chọn model theo UI
    selected = []

    for m in model_types:
        if condition in ["EO", "Both"] and "EO" in registry[m]:
            selected.append(registry[m]["EO"])
        if condition in ["EC", "Both"] and "EC" in registry[m]:
            selected.append(registry[m]["EC"])

    if not selected:
        st.warning("Không có mô hình nào trùng với lựa chọn!")
        st.stop()

    system = EEGSystem(selected, fs=500, epoch_sec=2.0, debug=False)

    temp = "temp_upload.csv"
    with open(temp, "wb") as f:
        f.write(file.getbuffer())

    with st.spinner("Đang xử lý EEG + dự đoán..."):
        t0 = time.time()
        rs, err = system.run(
            temp,
            age=age, gender=gender,
            education=education, sleep=sleep, well=well
        )
        dt = time.time() - t0

    os.remove(temp)

    if err:
        st.error(f"Pipeline error: {err}")
        st.stop()

    st.success(f"✔ Xử lý xong trong {dt:.2f}s")

    for r in rs["results"]:
        st.markdown(f"### 🔍 Model: `{r['model']}`")
        st.metric("Predict", r["pred_label"])

        proba = r["epoch_probs"]
        classes = r["classes"]

        epoch_preds = np.argmax(proba, axis=1)
        conf = (epoch_preds == r["pred_idx"]).mean()
        st.metric("Confidence", f"{conf*100:.2f}%")

        df = pd.DataFrame(proba, columns=classes)
        st.dataframe(df.style.highlight_max(axis=1))
