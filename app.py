# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Deteksi Fraud Kartu Kredit", layout="wide")
st.title("Deteksi Transaksi Fraud Kartu Kredit")
st.caption(
    "Upload CSV atau input manual + autofill dari baris dataset + tampil label asli dataset (jika ada). "
    "Prediksi memakai model terbaik dari notebook (LR/XGBoost) + threshold terbaik."
)

MODEL_PATH = "fraud_model.joblib"
THRESH_PATH = "best_threshold.txt"
DATA_PATH = "creditcard.csv"   # dataset untuk autofill & label asli (kalau ada)

# Label mapping (sesuai permintaan: non-fraud bukan normal)
LABEL_MAP = {0: "Non-Fraud (0)", 1: "Fraud (1)"}

# =========================
# Load model, threshold, data
# =========================
@st.cache_resource
def load_model(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"File model tidak ditemukan: {path}. "
            "Pastikan best_fraud_model.joblib ada di folder yang sama dengan app.py."
        )
    return joblib.load(p)

@st.cache_data
def load_threshold(path: str) -> float:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"File threshold tidak ditemukan: {path}. "
            "Pastikan best_threshold.txt ada di folder yang sama dengan app.py."
        )
    try:
        val = float(p.read_text().strip())
    except Exception:
        raise ValueError(f"Isi {path} tidak valid. Harus berupa angka float (contoh: 0.23).")

    if not (0.0 < val < 1.0):
        raise ValueError(f"Nilai threshold di {path} harus di antara 0 dan 1.")
    return val

@st.cache_data
def load_data_raw(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"File dataset tidak ditemukan: {path}. "
            "Pastikan creditcard.csv ada di folder yang sama dengan app.py."
        )
    return pd.read_csv(p)

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Gagal load model: {e}")
    st.stop()

try:
    threshold = load_threshold(THRESH_PATH)
except Exception as e:
    st.error(f"Gagal load threshold: {e}")
    st.stop()


# =========================
# Kolom fitur yang DIPAKAI model
# =========================
needed = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# =========================
# Load dataset raw untuk autofill + label asli (kalau ada)
# =========================
try:
    df_raw = load_data_raw(DATA_PATH)

    # label asli pembanding (jika ada kolom Class)
    if "Class" in df_raw.columns:
        y_true_all = df_raw["Class"].astype(int)
    else:
        y_true_all = None

    df_feat = df_raw.copy()
    if "Class" in df_feat.columns:
        df_feat = df_feat.drop(columns=["Class"])

    # validasi kolom sesuai model
    missing = [c for c in needed if c not in df_feat.columns]
    if missing:
        st.error(
            f"Dataset untuk autofill tidak sesuai kolom model.\n"
            f"Kolom yang hilang: {missing}\n\n"
            f"Pastikan dataset creditcard.csv punya kolom Time, V1..V28, Amount."
        )
        st.stop()

    # abaikan kolom ekstra kalau ada
    df_feat = df_feat[needed]

except Exception as e:
    st.error(
        f"Gagal load dataset dari '{DATA_PATH}'. Pastikan file CSV ada di folder yang sama dengan app.py.\n\n"
        f"Detail: {e}"
    )
    st.stop()

# =========================
# Helper: apply 1 row dataset -> session_state (autofill)
# =========================
def apply_row_to_session(row: pd.Series, columns: list[str]):
    for c in columns:
        if c not in row.index:
            continue
        v = row[c]
        if pd.isna(v):
            st.session_state[c] = 0.0
        else:
            try:
                st.session_state[c] = float(v)
            except Exception:
                st.session_state[c] = 0.0

# =========================
# Helper: predict df
# =========================
def predict_df(df_in: pd.DataFrame):
    # pastikan urutan kolom sesuai model
    df_in = df_in[needed].copy()

    # konversi numerik (robust)
    for c in df_in.columns:
        df_in[c] = pd.to_numeric(df_in[c], errors="coerce")
    df_in = df_in.fillna(0)

    # proba untuk kelas positif (fraud=1)
    proba = model.predict_proba(df_in)[:, 1]
    pred = (proba >= threshold).astype(int)
    return pred, proba

# =========================
# UI Tabs
# =========================
tab1, tab2 = st.tabs(["Upload CSV", "Input Manual (Autofill)"])

# =========================
# TAB 1: Upload CSV
# =========================
with tab1:
    st.subheader("Prediksi dari CSV")
    st.write(
        "Upload CSV berisi fitur (tanpa target). "
        "Jika kolom `Class` ada, akan ditampilkan sebagai label asli pembanding."
    )

    uploaded = st.file_uploader("Upload file CSV", type=["csv"])
    if uploaded is not None:
        df_in_raw = pd.read_csv(uploaded)

        # label asli untuk pembanding jika ada
        if "Class" in df_in_raw.columns:
            y_true_csv = pd.to_numeric(df_in_raw["Class"], errors="coerce").fillna(0).astype(int)
        else:
            y_true_csv = None

        df_in = df_in_raw.copy()
        if "Class" in df_in.columns:
            df_in = df_in.drop(columns=["Class"])

        # cek kolom
        missing = [c for c in needed if c not in df_in.columns]
        extra = [c for c in df_in.columns if c not in needed]
        if missing:
            st.error(f"Kolom CSV kurang: {missing}")
            st.stop()
        if extra:
            st.warning(f"Ada kolom ekstra, akan diabaikan: {extra}")
        df_in = df_in[needed]

        st.write("Preview data input:")
        st.dataframe(df_in.head(20), use_container_width=True)

        c1, c2 = st.columns([1, 2])
        with c1:
            st.info(f"Threshold aktif: **{threshold:.2f}**")
        with c2:
            st.write(
                "Prediksi kelas: **Fraud (1)** jika probabilitas >= threshold, "
                "selain itu **Non-Fraud (0)**."
            )

        if st.button("Prediksi (CSV)"):
            try:
                pred, proba = predict_df(df_in)

                out = df_in.copy()
                out["fraud_proba"] = proba
                out["fraud_pred"] = pred
                out["Label_Prediksi"] = out["fraud_pred"].map(LABEL_MAP)

                if y_true_csv is not None:
                    out["Label_Asli"] = y_true_csv.values
                    out["Label_Asli_Teks"] = out["Label_Asli"].map(LABEL_MAP)
                    out["Cocok_Label_Asli"] = (out["fraud_pred"].values == out["Label_Asli"].values)

                st.success("Selesai!")
                st.dataframe(out, use_container_width=True)

                st.download_button(
                    "Download hasil prediksi (CSV)",
                    data=out.to_csv(index=False).encode("utf-8"),
                    file_name="hasil_prediksi_fraud.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Gagal melakukan prediksi. Detail error: {e}")

# =========================
# TAB 2: Input Manual + Autofill
# =========================
with tab2:
    st.subheader("Input Manual (1 transaksi) — Autofill dari dataset")

    n = len(df_feat)
    if "selected_no" not in st.session_state:
        st.session_state["selected_no"] = 1

    c_pick, c_prev, c_next, c_fill = st.columns([3, 1, 1, 2])

    with c_pick:
        selected_no = st.number_input(
            "Pilih nomor data (1 = baris pertama)",
            min_value=1,
            max_value=n,
            value=int(st.session_state["selected_no"]),
            step=1,
        )
        st.session_state["selected_no"] = int(selected_no)

    with c_prev:
        st.write("")
        st.write("")
        if st.button("Prev"):
            st.session_state["selected_no"] = max(1, st.session_state["selected_no"] - 1)

    with c_next:
        st.write("")
        st.write("")
        if st.button("Next"):
            st.session_state["selected_no"] = min(n, st.session_state["selected_no"] + 1)

    with c_fill:
        st.write("")
        st.write("")
        if st.button("Isi Otomatis"):
            row0 = df_feat.iloc[st.session_state["selected_no"] - 1]
            apply_row_to_session(row0, needed)
            st.success(f"Form terisi dari dataset nomor {st.session_state['selected_no']}")

    # info label asli sebelum prediksi
    if y_true_all is not None:
        true_val = int(y_true_all.iloc[st.session_state["selected_no"] - 1])
        st.info(f"Label asli dataset (baris ini): **{LABEL_MAP.get(true_val, true_val)}**")
    else:
        st.warning("Kolom 'Class' tidak ditemukan di dataset, jadi label asli tidak bisa ditampilkan.")

    st.write("Semua fitur numerik (Time, V1..V28, Amount).")

    # ---- Form input manual ----
    with st.form("manual_form"):
        colA, colB, colC = st.columns(3)
        values = {}

        def num_input(name: str, container, step=0.01):
            default_val = st.session_state.get(name, 0.0)
            try:
                default_val = float(default_val)
            except Exception:
                default_val = 0.0
            return container.number_input(name, value=default_val, step=step, format="%.6f", key=name)

        for i, colname in enumerate(needed):
            container = colA if i % 3 == 0 else (colB if i % 3 == 1 else colC)
            if colname in ["Time", "Amount"]:
                values[colname] = num_input(colname, container, step=1.0)
            else:
                values[colname] = num_input(colname, container, step=0.01)

        submitted = st.form_submit_button("Prediksi (Manual)")

    if submitted:
        df_one = pd.DataFrame([values], columns=needed)

        try:
            pred, proba = predict_df(df_one)
            pred = int(pred[0])
            proba = float(proba[0])

            label = LABEL_MAP.get(pred, str(pred))
            st.success(f"Hasil prediksi model: **{label}**")
            st.write(f"Probabilitas: **{proba:.6f}**")

            st.write("Data input (dipakai model):")
            st.dataframe(df_one, use_container_width=True)

        except Exception as e:
            st.error("Gagal prediksi.\n\n" f"Detail: {e}")

st.divider()
st.caption(
    "Catatan: Prediksi kelas ditentukan dari probabilitas model dibandingkan threshold terbaik dari notebook."
)
