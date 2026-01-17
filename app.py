# ==========================================
# app.py — EEG Diagnostic System + TSV Auto-Fill
# ==========================================
from turtle import mode
import streamlit as st
import numpy as np
import pandas as pd
import time, os, re

from system.system import EEGSystem

# ============ MODELS ============
from predictor.cnn_tab_model import CNNTabModel
# from predictor.mlp_model import MLPECModel


# ==========================================
# STREAMLIT CONFIG
# ==========================================
st.set_page_config(page_title="EEG Diagnostic System",
                   layout="wide",
                   page_icon="🧠")

st.title("🧠 Multimodal EEG psychiatric disorder diagnostic systemm")
st.markdown("TDBRAIN TSV + EEG Multi-Condition + Multi-Model")
st.markdown("---")


# ==========================================
# HELPERS
# ==========================================
def detect_cond(fname: str):
    f = fname.lower()
    rules = [
        ("eo", "EO"),
        ("eyesopen", "EO"),
        ("open", "EO"),
        ("ec", "EC"),
        ("eyesclosed", "EC"),
        ("closed", "EC"),
    ]
    for k, v in rules:
        if k in f:
            return v
    return None


def extract_pid(fname: str):
    m = re.search(r"(sub-\d+)", fname.lower())
    return m.group(1) if m else None


# ==========================================
# TSV LOADER
# ==========================================
@st.cache_resource
def load_tsv():
    path = "data/TDBRAIN_participants_V2.tsv"
    if not os.path.exists(path):
        st.warning("⚠ TSV chưa tồn tại: data/TDBRAIN_participants_V2.tsv")
        return None
    df = pd.read_csv(path, sep="\t")
    df["participants_ID"] = df["participants_ID"].astype(str)
    return df


# ==========================================
# MODEL REGISTRY
# ==========================================
@st.cache_resource
def load_registry():
    registry = {
        "CNN-Tab": {
            "EO": CNNTabModel(condition="EO", debug=False),
            "EC": CNNTabModel(condition="EC", debug=False),
        },
        "CNN": {},
        "Tab": {},
        # "MLP": {"EC": MLPECModel()},
        "SVM": {},
        "XGBoost": {},
        "Random Forest": {},
    }
    return registry


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
    st.subheader("📈 Raw EEG Viewer")

    # chỉ xem file đầu (để nhẹ UI)
    f0 = files[0]
    df = pd.read_csv(f0)

    # assume all columns = channel
    channels = list(df.columns)

    # chọn kênh trực tiếp, default = tất cả hoặc 4 đầu
    default_sel = channels if len(channels) <= 4 else channels[:4]
    sel = st.multiselect("Chọn kênh", channels, default=default_sel)

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
participant_meta = None
pid = None
diagnosis = None

if files:
    pid = extract_pid(files[0].name)
    if pid and tsv is not None and pid in tsv["participants_ID"].values:
        row = tsv[tsv["participants_ID"] == pid].iloc[0]
        participant_meta = row
        diagnosis = row["indication"] if "indication" in row else None

        # st.info(f"📌 Detect participant: **{pid}**")
        if diagnosis and str(diagnosis) != "nan":
            # st.success(f"GT (TSV): **{diagnosis}**")
            pass

# ==========================================
# TSV DEFAULT → UI
# ==========================================
default_age = 25.0
default_gender = 1     # 1=Nam,0=Nữ
default_education = 12.0
default_sleep = 7.0
default_well = 0

if participant_meta is not None:
    pm = participant_meta

    # ---- age ----
    if "age" in pm and not pd.isna(pm["age"]):
        try:
            default_age = float(str(pm["age"]).replace(",", "."))
        except:
            pass

    # ---- education ----
    if "education" in pm and not pd.isna(pm["education"]):
        try:
            default_education = float(pm["education"])
        except:
            pass

    # ---- sleep ----
    if "sleep" in pm and not pd.isna(pm["sleep"]):
        try:
            default_sleep = float(pm["sleep"])
        except:
            pass

    # ---- well ----
    if "well" in pm and not pd.isna(pm["well"]):
        try:
            default_well = int(pm["well"])
        except:
            pass

    # ---- gender (TSV: 1=Male,0=Female; UI: 1=Nam,0=Nữ) ----
    if "gender" in pm and not pd.isna(pm["gender"]):
        try:
            default_gender = int(pm["gender"])
        except:
            pass


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.header("Thông tin bệnh nhân")

    age = st.number_input("Tuổi", 6.0, 100.0, default_age)

    gender = st.selectbox(
        "Giới tính",
        [1,0],   # 1=Nam, 0=Nữ
        index=[1,0].index(default_gender),
        format_func=lambda x: "Nam" if x==1 else "Nữ"
    )

    education = st.number_input("Học vấn (năm)", 0.0, 30.0, default_education)
    sleep = st.number_input("Giờ ngủ", 0.0, 24.0, default_sleep)

    well = st.selectbox(
        "Well-being",
        [-2,-1,0,1,2,3],
        index=[-2,-1,0,1,2,3].index(default_well)
    )

    st.markdown("---")
    model_types = st.multiselect(
        "Loại mô hình",
        ["CNN-Tab","CNN","Tab","MLP","SVM","XGBoost","Random Forest"],
        default=["CNN-Tab"]
    )

    condition = st.selectbox("Điều kiện EEG", ["EO","EC","Both"])


# ==========================================
# DIAGNOSIS PIPELINE
# ==========================================
if files and st.button("🚀 Chạy chẩn đoán", type="primary"):

    registry = load_registry()

    # group model by condition
    selected = {"EO": [], "EC": []}

    for m in model_types:
        if condition in ["EO", "Both"] and "EO" in registry[m]:
            selected["EO"].append(registry[m]["EO"])
        if condition in ["EC", "Both"] and "EC" in registry[m]:
            selected["EC"].append(registry[m]["EC"])

    if not selected["EO"] and not selected["EC"]:
        st.warning("Không có model phù hợp điều kiện đã chọn!")
        st.stop()

    # group file by condition
    cond_files = {"EO": [], "EC": []}

    for f in files:
        cond = detect_cond(f.name)
        if cond:
            cond_files[cond].append(f)
        else:
            st.warning(f"Không nhận diện được EO/EC từ tên file: {f.name}")

    if not cond_files["EO"] and not cond_files["EC"]:
        st.warning("Không có file EO hoặc EC!")
        st.stop()

    # system multi-run
    system = EEGSystem(models=selected, fs=500, epoch_sec=2.0, debug=False)

    with st.spinner("Đang xử lý EEG + dự đoán..."):
        t0 = time.time()
        rs, err = system.run_multi(
            cond_files=cond_files,
            age=age,
            gender=gender,
            education=education,
            sleep=sleep,
            well=well,
        )
        dt = time.time() - t0

    if err:
        st.error(f"Pipeline error: {err}")
        st.stop()

    st.success(f"✔ Xử lý xong trong {dt:.2f}s")

    cache = rs["cache"]

    for cond in cache:
        st.markdown(f"### 🧩 Debug features `{cond}`")
        X_psd = cache[cond]["psd"]
        X_coh = cache[cond]["coh"]
        st.markdown("#### 🔬 Coherence Heatmaps (band-averaged)")

        bands = ["delta","theta","alpha","beta","gamma"]

        import matplotlib.pyplot as plt

        # pick first epoch
        coh0 = X_coh[0]  # (ch,ch,B)
        ch = coh0.shape[0]

        fig, axes = plt.subplots(1, coh0.shape[-1], figsize=(3*coh0.shape[-1]+2, 4))
        for b in range(coh0.shape[-1]):
            ax = axes[b] if coh0.shape[-1]>1 else axes
            im = ax.imshow(coh0[:,:,b], vmin=0, vmax=1)
            ax.set_title(bands[b])
            fig.colorbar(im, ax=ax, fraction=0.046)

        st.pyplot(fig)

        st.write("X_psd:", X_psd.shape, "mean:", np.mean(X_psd), "min:", np.min(X_psd), "max:", np.max(X_psd))
        st.write("X_coh:", X_coh.shape, "mean:", np.mean(X_coh), "min:", np.min(X_coh), "max:", np.max(X_coh))


    # results
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

# =======================
# RESULT SUMMARY
# =======================
if 'rs' in locals() and rs is not None:
    summary_rows = []
    for r in rs["results"]:
        summary_rows.append({
            "Participant": pid,
            "Model": r["model"],
            "Predict": r["pred_label"],
            "GT (TSV)": diagnosis,
            "Confidence": f"{(r['epoch_probs'].argmax(axis=1)==r['pred_idx']).mean()*100:.1f}%",
        })

    st.subheader("📌 KẾT QUẢ DỰ ĐOÁN (SUMMARY)")
    st.dataframe(pd.DataFrame(summary_rows))
