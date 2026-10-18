from pathlib import Path

import pandas as pd
import streamlit as st

from knn_engine import prediksi_pasien, validasi_dataset

st.set_page_config(page_title="SPK Risiko Diabetes - KNN", page_icon="🩺", layout="centered")

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background:
      radial-gradient(circle at top left, rgba(255,255,255,0.45), transparent 28%),
      radial-gradient(circle at bottom right, rgba(255,255,255,0.35), transparent 24%),
      linear-gradient(135deg, #fff0f6 0%, #ffd9e8 42%, #f8c8dc 100%);
}
[data-testid="stHeader"] {background: rgba(0,0,0,0);}
.block-container {
    max-width: 680px;
    padding-top: 2rem;
    padding-bottom: 2.5rem;
}
.hero-box {
    background: rgba(255,255,255,0.72);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.55);
    box-shadow: 0 10px 32px rgba(152, 64, 108, 0.14);
    border-radius: 22px;
    padding: 1.2rem 1.2rem 1rem;
    margin-bottom: 1rem;
}
.hero-box h1 {
    margin: 0 0 0.35rem 0;
    font-size: 1.9rem !important;
    color: #7c3056;
}
.hero-sub {
    color: #8a5571;
    font-size: 0.98rem;
    margin-bottom: 0.2rem;
}
.form-card {
    background: rgba(255,255,255,0.84);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.58);
    border-radius: 22px;
    padding: 0.45rem 0.55rem 0.7rem;
    box-shadow: 0 12px 35px rgba(157, 72, 114, 0.13);
}
[data-testid="stForm"] {
    border: none;
    background: transparent;
    padding: 0.25rem 0.55rem 0.3rem;
}
.stNumberInput, .stSelectbox {margin-bottom: 0.25rem;}
.stButton button, [data-testid="stFormSubmitButton"] button {
    border-radius: 14px !important;
    height: 3rem;
    font-weight: 700;
    border: none;
    color: white !important;
    background: linear-gradient(90deg, #d9578d 0%, #c8457c 100%) !important;
    box-shadow: 0 10px 26px rgba(201, 69, 124, 0.28);
    transition: all 0.25s ease-in-out;
}
.stButton button:hover, [data-testid="stFormSubmitButton"] button:hover {
    transform: translateY(-2px) scale(1.01);
    box-shadow: 0 14px 28px rgba(201, 69, 124, 0.34);
    background: linear-gradient(90deg, #e26498 0%, #cf4d83 100%) !important;
}
.stButton button:active, [data-testid="stFormSubmitButton"] button:active {
    transform: scale(0.99);
}
.result-card {
    margin-top: 1rem;
    background: rgba(255,255,255,0.84);
    border-radius: 18px;
    padding: 0.9rem 1rem;
    border: 1px solid rgba(255,255,255,0.55);
    box-shadow: 0 12px 30px rgba(157, 72, 114, 0.12);
}
.note-box {
    margin-top: 1rem;
    color: #7f5269;
    font-size: 0.9rem;
    text-align: center;
}
@media (max-width: 640px) {
  .block-container {padding: 1.05rem .85rem 2rem;}
  .hero-box h1 {font-size: 1.55rem !important;}
  .hero-sub {font-size: 0.92rem;}
  .hero-box, .form-card {border-radius: 18px;}
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero-box">
        <h1>Prediksi Risiko Diabetes</h1>
        <div class="hero-sub">Sistem Pendukung Keputusan menggunakan metode K-Nearest Neighbors (KNN)</div>
    </div>
    """,
    unsafe_allow_html=True,
)

@st.cache_data
def muat_data():
    lokasi = Path(__file__).resolve().parent / "data" / "pasien_knn.csv"
    return validasi_dataset(pd.read_csv(lokasi))

try:
    dataset = muat_data()
except (OSError, ValueError, pd.errors.ParserError) as exc:
    st.error(f"Data pasien tidak dapat dibaca: {exc}")
    st.stop()

st.markdown('<div class="form-card">', unsafe_allow_html=True)
st.write("Masukkan data pasien:")
with st.form("input_pasien"):
    usia = st.number_input("Usia (tahun)", min_value=1, max_value=120, value=42, step=1)
    bmi = st.number_input("BMI", min_value=1.0, max_value=100.0, value=28.5, step=0.1, format="%.1f")
    glukosa = st.number_input("Glukosa", min_value=1, max_value=1000, value=130, step=1)
    tekanan = st.number_input("Tekanan darah", min_value=1, max_value=350, value=128, step=1)
    keluarga = st.selectbox("Riwayat diabetes dalam keluarga", ["Ada", "Tidak ada"])
    kirim = st.form_submit_button("Prediksi", type="primary", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

if kirim:
    data_baru = {
        "Usia": usia,
        "BMI": bmi,
        "Glukosa": glukosa,
        "Tekanan_Darah": tekanan,
        "Riwayat_Keluarga": 1 if keluarga == "Ada" else 0,
    }
    hasil = prediksi_pasien(dataset, data_baru, k=3)
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.subheader("Hasil Prediksi")
    if hasil.kelas == 1:
        st.error("**Risiko Tinggi**")
    else:
        st.success("**Risiko Rendah**")
    st.caption("Hasil ditentukan berdasarkan 3 data pasien terdekat (K = 3).")
    with st.expander("Lihat 3 tetangga terdekat"):
        tabel = hasil.tetangga[["Pasien", "Jarak", "Kategori"]].copy()
        tabel["Jarak"] = tabel["Jarak"].round(3)
        st.dataframe(tabel, hide_index=True, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="note-box">Catatan: Data berasal dari 10 pasien contoh untuk pembelajaran. Hasil bukan diagnosis medis.</div>',
    unsafe_allow_html=True,
)
