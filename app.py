# ==========================================
# app.py — EEG Diagnostic System + TSV + Clinical View
# ==========================================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_FLAGS"] = "--xla_cpu_use_xla=false"

import streamlit as st
import numpy as np
import pandas as pd
import time, os, re
import matplotlib.pyplot as plt

from system.system import EEGSystem
from predictor.cnn_tab_model import CNNTabModel
from predictor.cnn_model import CNNModel
from predictor.mlp_model import MLPModel
from predictor.svm_model import SVMModel
from predictor.tab_model import TabModel
from predictor.xgboost_model import XGBModel
from predictor.rf_model import RFModel


# ==========================================
# STREAMLIT CONFIG
# ==========================================
st.set_page_config(
    page_title="EEG Diagnostic System",
    layout="wide",
    page_icon="🧠"
)

# ==========================================
# HEADER W/ LEFT + RIGHT IMAGE
# ==========================================
# colL, colR = st.columns([1,1])

# with colL:
#     st.image("assets/logo_left.png", use_column_width=True)

# with colR:
#     st.image("assets/logo_right.png", use_column_width=True)

import base64

def img_to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

l = img_to_b64("assets/logo_left.png")
r = img_to_b64("assets/logo_right.png")

st.markdown(
    f"""
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <img src="data:image/png;base64,{l}" style="height:80px;">
        <img src="data:image/png;base64,{r}" style="height:80px;">
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<hr style='border:1px solid #444;'>", unsafe_allow_html=True)

st.markdown("""
<style>
header, hr {
    margin-top: 4px;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

# st.title("🧠 Multimodal EEG Psychiatric Diagnostic System")
# st.markdown("---")

st.markdown("""
<style>
.help-box {
    background:#f7faff;
    padding:10px 14px;
    border-radius:8px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("### 📘 Hướng dẫn sử dụng hệ thống")

colL, colR = st.columns([1.2, 1])

with colL:
    st.markdown("""
    **Dữ liệu đầu vào:**
    - Dữ liệu EEG .csv (chuẩn TDBRAIN)
    - Có thể upload EO / EC hoặc cả hai

    **Các bước thao tác:**
    1. Tải lên dữ liệu EEG ở mục `Upload EEG CSV files`
    2. Kiểm tra thông tin lâm sàng được tự động điền từ TSV (nếu có)
    3. Nhập bổ sung thông tin ở *Sidebar*
    4. Lựa chọn mô hình dự đoán và điều kiện EEG
    5. Nhấn **Chẩn đoán** để chạy pipeline
    6. Quan sát kết quả trên:
       - Tín hiệu EEG sau tiền xử lý 
       - Mật độ phổ công suất (PSD)  
       - Ma trận kết nối chức năng (coherence)  
       - Thống kê dự đoán mô hình
    """)

with colR:
    st.markdown("""
    **Lưu ý:**
    - Hệ thống hỗ trợ phân loại 3 nhóm lâm sàng:
      `Rối loạn tăng động giảm chú ý (ADHD) / Rối loạn trầm cảm (MDD) / Rối loạn nhận thức chủ quan (SMC)`
    - PSD và Coherence được tính theo **từng epoch**
    - Cho phép lựa chọn điều kiện đo EO / EC hoặc đồng thời
    - Cho phép chạy nhiều mô hình dự đoán song song
    - Hỗ trợ cơ chế tự động điền thông tin lâm sàng từ file TSV
    """)



# ==========================================
# HELPERS
# ==========================================
def detect_cond(fname: str):
    f = fname.lower()
    rules = [
        ("eo","EO"),("eyesopen","EO"),("open","EO"),
        ("ec","EC"),("eyesclosed","EC"),("closed","EC"),
    ]
    for k,v in rules:
        if k in f: return v
    return None


def extract_pid(fname: str):
    m = re.search(r"(sub-\d+)", fname.lower())
    return m.group(1) if m else None


def safe_default_channels(ch, k=4):
    return ch[:k] if len(ch) > k else ch


# ==========================================
# TSV LOADER
# ==========================================
@st.cache_resource
def load_tsv():
    path = "data/TDBRAIN_participants_V2.tsv"
    if not os.path.exists(path):
        st.warning("⚠ Missing TSV: data/TDBRAIN_participants_V2.tsv")
        return None
    df = pd.read_csv(path, sep="\t")
    df["participants_ID"] = df["participants_ID"].astype(str)
    return df


# ==========================================
# MODEL REGISTRY
# ==========================================
from functools import lru_cache

# ---- cache only CPU models (safe) ----
@lru_cache(maxsize=16)
def get_cpu_model(model_type, cond):
    if model_type == "SVM": return SVMModel(condition=cond)
    if model_type == "XGBoost": return XGBModel(condition=cond)
    if model_type == "Random Forest": return RFModel(condition=cond)
    raise ValueError("Not CPU model")

# ---- do NOT cache TF models ----
def get_model(model_type, cond):
    if model_type == "CNN-Tab": return CNNTabModel(condition=cond)
    if model_type == "CNN":     return CNNModel(condition=cond)
    if model_type == "Tab":     return TabModel(condition=cond)
    if model_type == "MLP":     return MLPModel(condition=cond)
    # CPU:
    return get_cpu_model(model_type, cond)

# ==========================================
# FILE UPLOAD
# ==========================================
files = st.file_uploader(
    "📤 Upload EEG CSV files (EO/EC)",
    accept_multiple_files=True,
    type=["csv"]
)


# ==========================================
# RAW EEG VIEWER (MULTI-CHANNEL)
# ==========================================
if files:
    st.subheader("📈 Raw EEG Viewer (Pre-Diagnosis)")

    import io
    uploaded = files[0]
    content = uploaded.read()
    uploaded.seek(0)

    try:
        df = pd.read_csv(io.BytesIO(content), sep=None, engine="python")
    except Exception as e:
        st.error(f"Không đọc được CSV: {e}")
        st.stop()

    ch_raw = list(df.columns)

    # assume all columns = channel
    channels = list(df.columns)

    # init session default channels
    if "raw_sel" not in st.session_state:
        st.session_state.raw_sel = safe_default_channels(ch_raw)

    sel = st.multiselect(
        "Chọn kênh (RAW)",
        ch_raw,
        default=st.session_state.raw_sel,
        key="raw_sel"
    )


    # fixed sampling rate
    fs = 500

    # viewport
    N = len(df)
    win = min(60000, N)
    start = 0

    t = np.arange(N) / fs # time vector

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(sel), 1, figsize=(12, 2.2*len(sel)), sharex=True)

    if len(sel) == 1:
        axes = [axes]

    for i, c in enumerate(sel):
        y = df[c].values

        axes[i].plot(t[start:start+win], y[start:start+win])
        axes[i].set_xlabel("Time (s)")

        axes[i].set_ylabel(c)
        axes[i].grid(True)

    st.pyplot(fig)
    st.markdown("---")



# ==========================================
# TSV AUTO-FILL
# ==========================================
tsv = load_tsv()
pid = None
diagnosis = None
participant_meta = None

if files:
    pid = extract_pid(files[0].name)
    if pid and tsv is not None and pid in tsv["participants_ID"].values:
        row = tsv[tsv["participants_ID"] == pid].iloc[0]
        participant_meta = row
        diagnosis = row.get("indication", None)


# ==========================================
# DEFAULT CLINICAL VALUES
# ==========================================
default_age = 25.0
default_gender = 1
default_education = 12.0
default_sleep = 7.0
default_well = 0

if participant_meta is not None:
    def sf(v,fb):
        try: return float(str(v).replace(",",".")) 
        except: return fb
    def si(v,fb):
        try: return int(v) 
        except: return fb

    pm = participant_meta
    default_age = sf(pm.get("age",default_age), default_age)
    default_education = sf(pm.get("education",default_education), default_education)
    default_sleep = sf(pm.get("sleep",default_sleep), default_sleep)
    default_well = si(pm.get("well",default_well), default_well)
    default_gender = si(pm.get("gender",default_gender), default_gender)


# ==========================================
# SIDEBAR (Clinical Inputs)
# ==========================================
with st.sidebar:
    st.header("Thông tin bệnh nhân")

    age = st.number_input("Tuổi", 6.0, 100.0, default_age)
    gender = st.selectbox("Giới tính", [1,0],
                          index=[1,0].index(default_gender),
                          format_func=lambda x: "Nam" if x==1 else "Nữ")

    education = st.number_input("Học vấn (năm)", 0.0, 30.0, default_education)
    sleep = st.number_input("Giờ ngủ", 0.0, 24.0, default_sleep)
    well = st.selectbox("Well-being", [-2,-1,0,1,2,3],
                        index=[-2,-1,0,1,2,3].index(default_well))

    st.markdown("---")

    model_types = st.multiselect("Loại mô hình", 
                                 ["CNN-Tab", "CNN", "Tab", "MLP", "SVM", "XGBoost", "Random Forest"],
                                    default=["CNN-Tab"])
    condition = st.selectbox("Điều kiện EEG", ["EO","EC","Both"])

run_btn = st.button("🚀 Chẩn đoán", type="primary")


# ==========================================
# SESSION_STATE INIT
# ==========================================
if "rs" not in st.session_state:
    st.session_state.rs = None
    st.session_state.cache = None
    st.session_state.ch_names = None
    st.session_state.pid = None
    st.session_state.diagnosis = None


# ==========================================
# PIPELINE EXECUTE
# ==========================================
if files and run_btn:
    # try = load_registry()
    # selected = {"EO": [], "EC": []}

    # for m in model_types:
    #     if condition in ["EO","Both"] and "EO" in registry[m]: selected["EO"].append(registry[m]["EO"])
    #     if condition in ["EC","Both"] and "EC" in registry[m]: selected["EC"].append(registry[m]["EC"])

    selected = {"EO": [], "EC": []}
    for m in model_types:
        if condition in ["EO","Both"]:
            selected["EO"].append(get_model(m,"EO"))
        if condition in ["EC","Both"]:
            selected["EC"].append(get_model(m,"EC"))


    cond_files = {"EO": [], "EC": []}
    for f in files:
        cd = detect_cond(f.name)
        if cd: cond_files[cd].append(f)

    system = EEGSystem(models=selected, fs=500, epoch_sec=2.0, debug=True)

    with st.spinner("⏳ Đang xử lý EEG + chạy dự đoán..."):
        t0 = time.time()
        rs, err = system.run_multi(
            cond_files=cond_files,
            age=age, gender=gender,
            education=education,
            sleep=sleep, well=well   
        )
        dt = time.time()-t0

    if err:
        st.error(f"Pipeline error: {err}")
        st.stop()

    st.success(f"✔ Pipeline hoàn tất trong {dt:.2f}s")

    st.session_state.rs = rs
    st.session_state.cache = rs["cache"]
    st.session_state.pid = pid
    st.session_state.diagnosis = diagnosis

    # read channel labels from the first file
    f0 = pd.read_csv(files[0])
    st.session_state.ch_names = list(f0.columns)

# ==========================================
# CLINICAL VIEWER + PREDICTION (No rerun)
# ==========================================
if st.session_state.rs is not None:
    rs = st.session_state.rs
    cache = st.session_state.cache
    pid = st.session_state.pid
    diagnosis = st.session_state.diagnosis
    ch_names = st.session_state.ch_names

    st.markdown("---")
    st.markdown("## 🎧 EEG Viewer (Clinical)")

    cond_list = list(cache.keys())
    cond_sel = st.selectbox("Điều kiện", cond_list)

    seg = cache[cond_sel]["seg"]     # (E,C,T)
    X_coh = cache[cond_sel]["coh"]   # (E,C,C,B)
    fs = 500
    E,C,T = seg.shape

    tabs = st.tabs(["Waveform","PSD","Coherence"])

    with tabs[0]:
        concat = np.concatenate(seg, axis=1)
        t = np.arange(concat.shape[1])/fs
        sel = st.multiselect("Chọn kênh", ch_names, default=safe_default_channels(ch_names))
        fig, axes = plt.subplots(len(sel),1,figsize=(12,2.2*len(sel)),sharex=True)
        if len(sel)==1: axes=[axes]
        for i,c in enumerate(sel):
            idx = ch_names.index(c)
            axes[i].plot(t, concat[idx], linewidth=0.7)
            axes[i].set_ylabel(c); axes[i].grid()
        axes[-1].set_xlabel("Time (s)")
        st.pyplot(fig)

    with tabs[1]:
        from scipy.signal import welch
        psd_list=[]
        for ep in seg:
            f,Pxx=welch(ep,fs=fs,nperseg=int(fs*2))
            psd_list.append(Pxx)
        psd_arr=np.stack(psd_list,0)
        sel = st.multiselect("Channel (PSD)", ch_names, default=safe_default_channels(ch_names))
        fig, ax = plt.subplots(figsize=(10,4))
        for c in sel:
            idx = ch_names.index(c)
            mean_psd = psd_arr[:,idx].mean(0)
            for k in range(psd_arr.shape[0]):
                ax.semilogy(f, psd_arr[k,idx], alpha=0.15, linewidth=0.7)
            ax.semilogy(f, mean_psd, linewidth=2, label=f"{c} (mean)")
        ax.set_xlim(1,45); ax.grid(); ax.legend()
        st.pyplot(fig)

    with tabs[2]:
        st.markdown("Coherence (Mean across epochs)")
        mean_coh = X_coh.mean(0)
        bands=["delta","theta","alpha","beta","gamma"]
        B=mean_coh.shape[-1]
        fig,axes=plt.subplots(1,B,figsize=(3*B+2,4))
        for b in range(B):
            im=axes[b].imshow(mean_coh[:,:,b],vmin=0,vmax=1)
            axes[b].set_title(bands[b])
            fig.colorbar(im,ax=axes[b], fraction=0.046)
        st.pyplot(fig)

    st.markdown("---")
    st.markdown("## 🧠 Prediction Summary")

    summary=[]
    for r in rs["results"]:
        st.markdown(f"### 🔍 Model: `{r['model']}`")
        st.metric("Predict", r["pred_label"])
        proba=r["epoch_probs"]
        classes=r["classes"]
        epoch_preds = np.argmax(proba, 1)
        hit = (epoch_preds == r["pred_idx"]).sum()
        E = len(epoch_preds)
        conf = hit / E
        st.metric("Epoch Hits", f"{hit}/{E} (Confidence: {conf:.2%})")
        dfp=pd.DataFrame(proba,columns=classes)
        st.dataframe(dfp.style.highlight_max(axis=1), use_container_width=True)

        summary.append({
            "Participant": pid,
            "Model": r["model"],
            "Predict": r["pred_label"],
            "GT (TSV)": diagnosis,
            "Epoch Hits": f"{hit}/{E}"        
            })

    st.subheader("📌 FINAL SUMMARY")
    st.dataframe(pd.DataFrame(summary), use_container_width=True)
