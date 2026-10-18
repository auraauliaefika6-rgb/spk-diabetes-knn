from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SPK Risiko Diabetes | KNN", page_icon="🌸", layout="centered")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
 background: radial-gradient(circle at 10% 5%, #ffffffad, transparent 30%),
 radial-gradient(circle at 95% 90%, #ffffff88, transparent 35%),
 linear-gradient(135deg,#fff2f7 0%,#ffdeeb 50%,#f7c8dc 100%);
}
[data-testid="stHeader"] {background: transparent}
.block-container {max-width:750px;padding-top:2rem;padding-bottom:3rem}
.hero {background:#ffffffc9;border:1px solid #ffffffea;border-radius:22px;
 padding:22px 25px;box-shadow:0 12px 34px #98406c18;margin-bottom:22px}
.hero h1 {font-size:1.9rem;color:#812f58;margin:0 0 6px}
.hero p {color:#87516d;margin:0}
[data-testid="stForm"] {background:#ffffffd9;border:1px solid white;
 border-radius:22px;padding:23px;box-shadow:0 12px 34px #98406c1b}
[data-testid="stFormSubmitButton"] button {
 background:linear-gradient(110deg,#df6499,#c63e7a)!important;color:white!important;
 border:0!important;border-radius:13px!important;font-weight:750!important;
 min-height:49px;box-shadow:0 8px 20px #c63e7a50;
 transition:transform .22s ease,box-shadow .22s ease,filter .22s ease}
[data-testid="stFormSubmitButton"] button:hover {
 transform:translateY(-3px);box-shadow:0 13px 25px #c63e7a65;filter:brightness(1.06)}
[data-testid="stFormSubmitButton"] button:active {transform:scale(.98)}
[data-testid="stFormSubmitButton"] button:focus-visible {outline:3px solid #8f215a;outline-offset:3px}
.result {background:#ffffffdc;border:1px solid #fff;border-radius:18px;
 padding:18px 23px;margin:22px 0 14px;box-shadow:0 9px 28px #98406c15}
.result h2 {color:#812f58;margin:0 0 10px;font-size:1.35rem}
.risk-high {background:#fff0f1;border-left:5px solid #e34c61;color:#9b2439;
 border-radius:12px;padding:14px 17px;font-size:1.2rem;font-weight:750}
.risk-low {background:#ecfaf2;border-left:5px solid #289766;color:#176644;
 border-radius:12px;padding:14px 17px;font-size:1.2rem;font-weight:750}
.small-note {color:#76566a;font-size:.87rem;text-align:center;margin-top:22px}
@media(max-width:650px) {
 .block-container{padding:1rem .8rem 2rem}
 .hero{padding:18px}.hero h1{font-size:1.48rem}
 [data-testid="stForm"]{padding:14px}
}
@media(prefers-reduced-motion:reduce) {
 [data-testid="stFormSubmitButton"] button{transition:none!important}
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    path = Path(__file__).resolve().parent / "data" / "pasien_knn.csv"
    df = pd.read_csv(path)
    cols = ["Pasien","Usia","BMI","Glukosa","Tekanan_Darah","Riwayat_Keluarga","Kelas"]
    if not set(cols).issubset(df.columns):
        raise ValueError("Kolom dataset tidak lengkap.")
    df = df[cols].copy()
    for col in cols[1:]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    if df.isna().any().any() or not df["Kelas"].isin([0,1]).all() or not df["Riwayat_Keluarga"].isin([0,1]).all():
        raise ValueError("Dataset berisi nilai tidak valid.")
    if len(df) < 3:
        raise ValueError("Dataset minimal 3 pasien.")
    return df

def calculate_knn(df, patient, k=3):
    """Jarak kuadrat Euclidean pada fitur asli, seperti contoh referensi."""
    cols = ["Usia","BMI","Glukosa","Tekanan_Darah","Riwayat_Keluarga"]
    matrix = df[cols].to_numpy(dtype=float)
    target = np.array([patient[col] for col in cols],dtype=float)
    d2 = np.sum((matrix-target)**2,axis=1)
    distances = df[["Pasien","Kelas"]].copy()
    distances["Jarak²"] = np.round(d2,4)
    distances["_raw"] = d2
    distances = distances.sort_values(["_raw","Pasien"],kind="stable").reset_index(drop=True)
    distances.insert(0,"Peringkat",np.arange(1,len(distances)+1))
    distances["Kelas"] = distances["Kelas"].map({0:"Rendah",1:"Tinggi"})
    distances["Dipakai"] = ["Ya" if i<k else "-" for i in range(len(distances))]
    neighbors = distances.head(k).copy()
    tinggi = int((neighbors["Kelas"]=="Tinggi").sum())
    rendah = k-tinggi
    return ("Tinggi" if tinggi>rendah else "Rendah"),tinggi,rendah,neighbors,distances

st.markdown('<div class="hero"><h1>Prediksi Risiko Diabetes</h1><p>Sistem Pendukung Keputusan menggunakan metode K-Nearest Neighbors (KNN)</p></div>',unsafe_allow_html=True)
try:
    df = load_data()
except (OSError, ValueError, pd.errors.ParserError) as exc:
    st.error(f"Dataset tidak dapat dimuat: {exc}")
    st.stop()

with st.form("form_prediksi"):
    st.markdown("**Masukkan data pasien:**")
    usia = st.number_input("Usia (tahun)",min_value=1,max_value=120,value=42)
    bmi = st.number_input("BMI",min_value=1.0,max_value=100.0,value=28.5,step=0.1,format="%.1f")
    glukosa = st.number_input("Glukosa",min_value=1,max_value=1000,value=130)
    tekanan = st.number_input("Tekanan darah",min_value=1,max_value=350,value=128)
    keluarga = st.selectbox("Riwayat diabetes dalam keluarga",["Ada","Tidak ada"])
    submitted = st.form_submit_button("Prediksi",type="primary",use_container_width=True)

if submitted:
    patient = dict(Usia=usia,BMI=bmi,Glukosa=glukosa,Tekanan_Darah=tekanan,
                   Riwayat_Keluarga=int(keluarga=="Ada"))
    result, tinggi, rendah, neighbors, distances = calculate_knn(df,patient)
    st.markdown('<div class="result"><h2>Hasil Prediksi</h2><div class="risk-'+
                ('high' if result=="Tinggi" else 'low')+'">Risiko '+result+
                '</div></div>',unsafe_allow_html=True)
    st.write(f"**Voting KNN (K = 3):** {tinggi} dari 3 tetangga berisiko Tinggi dan {rendah} berisiko Rendah.")
    st.progress(tinggi/3,text=f"Tinggi {tinggi}  •  Rendah {rendah}")
    st.caption("Kelas dengan suara terbanyak menjadi hasil prediksi.")
    st.markdown("#### 3 tetangga terdekat")
    st.dataframe(neighbors[["Peringkat","Pasien","Kelas","Jarak²"]],hide_index=True,use_container_width=True)
    with st.expander("Lihat jarak ke seluruh data"):
        st.dataframe(distances[["Peringkat","Pasien","Kelas","Jarak²","Dipakai"]],
                     hide_index=True,use_container_width=True)
        st.caption("Jarak² = jumlah kuadrat selisih lima atribut tanpa standardisasi.")
        st.write("Contoh perhitungan untuk tetangga terdekat:")
        closest = df.loc[df["Pasien"]==neighbors.iloc[0]["Pasien"]].iloc[0]
        parts = [f"({patient[c]} − {closest[c]})²" for c in
                 ["Usia","BMI","Glukosa","Tekanan_Darah","Riwayat_Keluarga"]]
        st.code(" + ".join(parts)+f" = {neighbors.iloc[0]['Jarak²']:g}",language=None)

st.markdown('<div class="small-note">Dataset 10 pasien hanya untuk pembelajaran. Hasil bukan diagnosis medis.</div>',unsafe_allow_html=True)
