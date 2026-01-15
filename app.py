import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from utils.predict_logic import PredictorSystem

st.set_page_config(page_title="EEG Diagnostic System", layout="wide", page_icon="🧠")


@st.cache_resource
def get_system():
    return PredictorSystem()


# Init backend
try:
    with st.spinner("Đang tải mô hình và modules EEG..."):
        system = get_system()
    st.toast("Hệ thống đã sẵn sàng!", icon="✅")
except Exception as e:
    st.error(f"Lỗi khởi động: {e}")
    st.stop()


# Sidebar
with st.sidebar:
    st.title("Thông tin bệnh nhân")
    age = st.number_input("Tuổi (Age)", 6.0, 100.0, 25.0)
    gender = st.selectbox("Giới tính", [1, 0], format_func=lambda x: "Nam" if x == 1 else "Nữ")
    education = st.number_input("Học vấn (năm)", 0.0, 30.0, 12.0)
    sleep = st.number_input("Giờ ngủ/ngày", 0.0, 24.0, 7.0)
    well = st.selectbox("Chỉ số Well-being", [-2, -1, 0, 1, 2, 3], index=2)


# Main UI
st.title("🧠 Phân loại EEG đa lớp (TabTransformer + CNN + PSD + COH)")
st.markdown("---")

uploaded_file = st.file_uploader("📤 Tải lên file CSV EEG thô (raw EEG)", type=["csv"])

if uploaded_file and st.button("🚀 Chạy Chẩn đoán", type="primary"):
    with st.spinner("Đang xử lý EEG + trích chọn đặc trưng..."):
        temp_dir = "utils/temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, "temp_upload.csv")

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        start = time.time()
        try:
            probs, classes = system.process_and_predict(temp_path, age, gender, education, sleep, well)
            duration = time.time() - start

            if probs is None:
                st.error(classes)
            else:
                st.success(f"✔ Hoàn thành trong {duration:.2f}s")

                avg_probs = np.mean(probs, axis=0)
                best_idx = np.argmax(avg_probs)

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("Kết quả dự đoán (Subject-level)", classes[best_idx],
                              f"{avg_probs[best_idx]*100:.2f}%")

                with col2:
                    df_chart = pd.DataFrame({"Class": classes, "Probability": avg_probs})
                    st.bar_chart(df_chart, x="Class", y="Probability")

                st.subheader("Chi tiết từng Epoch (2s)")
                st.dataframe(pd.DataFrame(probs, columns=classes).style.highlight_max(axis=1))

        except Exception as e:
            st.error(f"Lỗi pipeline: {e}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
