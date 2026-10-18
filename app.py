"""Tampilan sederhana untuk studi kasus KNN, kompatibel dengan knn_engine.py."""
from pathlib import Path

import pandas as pd
import streamlit as st

from knn_engine import LABELS, prediksi_pasien, validasi_dataset

st.set_page_config(page_title="SPK Risiko Diabetes - KNN", page_icon="🩺", layout="centered")

st.markdown("""
<style>
.block-container {max-width: 650px; padding-top: 2rem; padding-bottom: 2.5rem;}
h1 {font-size: 1.75rem !important;}
[data-testid="stForm"] {border: 1px solid #e5e7eb; border-radius: 12px; padding: 1.2rem;}
.stButton button, [data-testid="stFormSubmitButton"] button {border-radius: 8px;}
@media (max-width: 640px) {
  .block-container {padding: 1.1rem .9rem 2rem;}
  h1 {font-size: 1.48rem !important;}
}
</style>
""", unsafe_allow_html=True)

st.title("Prediksi Risiko Diabetes")
st.caption("Sistem Pendukung Keputusan menggunakan metode K-Nearest Neighbors (KNN)")

@st.cache_data
def muat_data():
    lokasi = Path(__file__).resolve().parent / "data" / "pasien_knn.csv"
    return validasi_dataset(pd.read_csv(lokasi))

try:
    dataset = muat_data()
except (OSError, ValueError, pd.errors.ParserError) as exc:
    st.error(f"Data pasien tidak dapat dibaca: {exc}")
    st.stop()

st.write("Masukkan data pasien:")
with st.form("input_pasien"):
    usia = st.number_input("Usia (tahun)", min_value=1, max_value=120, value=42, step=1)
    bmi = st.number_input("BMI", min_value=1.0, max_value=100.0, value=28.5, step=0.1, format="%.1f")
    glukosa = st.number_input("Glukosa", min_value=1, max_value=1000, value=130, step=1)
    tekanan = st.number_input("Tekanan darah", min_value=1, max_value=350, value=128, step=1)
    keluarga = st.selectbox("Riwayat diabetes dalam keluarga", ["Ada", "Tidak ada"])
    kirim = st.form_submit_button("Prediksi", type="primary", use_container_width=True)

if kirim:
    data_baru = {
        "Usia": usia,
        "BMI": bmi,
        "Glukosa": glukosa,
        "Tekanan_Darah": tekanan,
        "Riwayat_Keluarga": 1 if keluarga == "Ada" else 0,
    }
    # K=3 tetap sesuai contoh studi kasus, dengan standardisasi fitur.
    hasil = prediksi_pasien(dataset, data_baru, k=3)
    st.subheader("Hasil Prediksi")
    if hasil.kelas == 1:
        st.error("**Risiko Tinggi**")
    else:
        st.success("**Risiko Rendah**")
    st.write("Klasifikasi berdasarkan 3 data pasien terdekat (K = 3).")
    with st.expander("Lihat 3 tetangga terdekat"):
        tabel = hasil.tetangga[["Pasien", "Jarak", "Kategori"]].copy()
        tabel["Jarak"] = tabel["Jarak"].round(3)
        st.dataframe(tabel, hide_index=True, use_container_width=True)

st.caption("Catatan: Data berasal dari 10 pasien contoh untuk pembelajaran. Hasil bukan diagnosis medis.")
