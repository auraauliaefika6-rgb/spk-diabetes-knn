"""Mesin klasifikasi KNN edukatif berdasarkan Studi_Kasus_KNN.pdf (10 contoh)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["Usia", "BMI", "Glukosa", "Tekanan_Darah", "Riwayat_Keluarga"]
NEEDED = ["Pasien", *FEATURES, "Kelas"]
LABELS = {0: "Risiko Rendah", 1: "Risiko Tinggi"}


@dataclass
class HasilPrediksi:
    kelas: int
    proporsi_rendah: float
    proporsi_tinggi: float
    tetangga: pd.DataFrame


def validasi_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Validasi kolom dan batas logis minimum; bukan validasi klinis."""
    missing = [x for x in NEEDED if x not in data.columns]
    if missing:
        raise ValueError("Kolom CSV belum lengkap: " + ", ".join(missing))
    df = data[NEEDED].copy()
    if df.isna().any().any():
        raise ValueError("CSV mengandung data kosong. Lengkapi semua kolom.")
    if len(df) < 4:
        raise ValueError("Diperlukan sedikitnya 4 baris data.")
    df["Pasien"] = df["Pasien"].astype(str).str.strip()
    if df["Pasien"].eq("").any() or df["Pasien"].duplicated().any():
        raise ValueError("Kolom Pasien wajib berisi ID unik yang tidak kosong.")
    for c in FEATURES + ["Kelas"]:
        try:
            df[c] = pd.to_numeric(df[c], errors="raise")
        except (ValueError, TypeError):
            raise ValueError(f"Kolom {c} harus berisi angka.") from None
        if not np.isfinite(df[c].to_numpy(dtype=float)).all():
            raise ValueError(f"Kolom {c} mengandung nilai tidak valid.")
    if not df["Kelas"].isin([0, 1]).all() or not df["Riwayat_Keluarga"].isin([0, 1]).all():
        raise ValueError("Kelas dan Riwayat_Keluarga hanya boleh 0 atau 1.")
    if set(df["Kelas"].unique()) != {0, 1}:
        raise ValueError("Dataset harus mempunyai dua kategori kelas: 0 dan 1.")
    if (df[["Usia", "BMI", "Glukosa", "Tekanan_Darah"]] <= 0).any().any():
        raise ValueError("Usia, BMI, glukosa, dan tekanan darah harus lebih besar dari 0.")
    return df.reset_index(drop=True)


def kandidat_k(n: int) -> list[int]:
    """Ganjil dan bisa dipakai juga untuk leave-one-out (n-1 data latih)."""
    return [k for k in range(1, min(21, n - 1) + 1, 2)]


def buat_model(k: int) -> Pipeline:
    return Pipeline([
        ("standardisasi", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k, weights="uniform", metric="euclidean")),
    ])


def prediksi_pasien(df: pd.DataFrame, values: dict[str, float], k: int) -> HasilPrediksi:
    if k not in kandidat_k(len(df)):
        raise ValueError("Nilai K tidak valid untuk jumlah data tersedia.")
    X, y = df[FEATURES], df["Kelas"].astype(int)
    pasien = pd.DataFrame([{c: values[c] for c in FEATURES}], columns=FEATURES)
    pipe = buat_model(k).fit(X, y)
    kelas = int(pipe.predict(pasien)[0])
    peluang = dict(zip(pipe.named_steps["knn"].classes_.tolist(), pipe.predict_proba(pasien)[0].tolist()))
    data_baru_skala = pipe.named_steps["standardisasi"].transform(pasien)
    jarak, posisi = pipe.named_steps["knn"].kneighbors(data_baru_skala, n_neighbors=k)
    tabel = df.iloc[posisi[0]][NEEDED].copy()
    tabel.insert(1, "Jarak", jarak[0])
    tabel["Kategori"] = tabel["Kelas"].map(LABELS)
    tabel = tabel.drop(columns=["Kelas"]).reset_index(drop=True)
    tabel.insert(0, "Urutan", np.arange(1, k + 1))
    return HasilPrediksi(
        kelas=kelas,
        proporsi_rendah=float(peluang.get(0, 0)),
        proporsi_tinggi=float(peluang.get(1, 0)),
        tetangga=tabel,
    )


def evaluasi_loocv(df: pd.DataFrame, daftar_k: list[int]) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    """Refit standardisasi DI SETIAP fold, sehingga tidak bocor dari baris yang diuji."""
    X, y = df[FEATURES], df["Kelas"].astype(int)
    rows, cm_per_k = [], {}
    for k in daftar_k:
        prediksi = np.zeros(len(y), dtype=int)
        for latih, uji in LeaveOneOut().split(X):
            pipe = buat_model(k).fit(X.iloc[latih], y.iloc[latih])
            prediksi[uji[0]] = int(pipe.predict(X.iloc[uji])[0])
        a = accuracy_score(y, prediksi)
        ba = balanced_accuracy_score(y, prediksi)
        f1 = f1_score(y, prediksi, average="macro", zero_division=0)
        rows.append({"K": k, "Benar": int((prediksi == y.to_numpy()).sum()),
                     "Jumlah": len(y), "Accuracy": round(a, 4),
                     "Balanced Accuracy": round(ba, 4), "Macro F1": round(f1, 4)})
        cm_per_k[k] = pd.DataFrame(confusion_matrix(y, prediksi, labels=[0, 1]),
                                  index=["Aktual: Rendah", "Aktual: Tinggi"],
                                  columns=["Prediksi: Rendah", "Prediksi: Tinggi"])
    return pd.DataFrame(rows), cm_per_k
