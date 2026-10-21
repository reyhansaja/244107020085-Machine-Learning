# JS02 — Pemahaman Data dan Pra Pengolahan Data

Praktikum JTI Polinema 2025/2026, diimplementasikan sebagai script Python yang dapat dijalankan ulang.

## Struktur

```
data/                 dataset mentah (unduhan asli GitBook + Wikimedia)
praktikum-1/          EDA dataset Titanic
praktikum-2/          imputasi missing value
praktikum-3/          seleksi fitur, label encoding, standardisasi
praktikum-4/          pra pengolahan citra (Lenna)
tugas/                Wisconsin Breast Cancer
<praktikum>/output/   grafik PNG + tabel CSV hasil
```

## Cara jalan

```bash
C:/code/pemmes26/.venv/Scripts/python.exe praktikum-1/eda_titanic.ipynb
C:/code/pemmes26/.venv/Scripts/python.exe praktikum-2/imputasi_titanic.ipynb
C:/code/pemmes26/.venv/Scripts/python.exe praktikum-3/encoding_scaling.ipynb
C:/code/pemmes26/.venv/Scripts/python.exe praktikum-4/citra_lenna.ipynb
C:/code/pemmes26/.venv/Scripts/python.exe tugas/wbc_preprocessing.ipynb
```

Butuh: pandas, numpy, matplotlib, seaborn, scikit-learn, opencv-python.
Semua path dihitung dari `Path(__file__).resolve().parent`, jadi bisa dijalankan dari direktori mana pun.
Backend matplotlib = Agg (tanpa `plt.show()`), grafik disimpan ke `output/`.

## Sumber data

| File | Sumber |
|---|---|
| `Titanic-Dataset.csv` | GitBook JS02 Praktikum 1 (`/files/kISpryGBZublDrlIF2UX` → `3041032130-files.gitbook.io`) |
| `Titanic-Dataset-fixed.csv` | GitBook JS02 Praktikum 3 (`/files/YQ92Cr86kFtcX4D1dIfF`) |
| `wbc.csv` | GitBook JS02 Tugas Praktikum (`/files/8sVT5JNYn0W4dFgqeaq4`) |
| `Lenna.png` | Wikimedia Commons `Lenna_(test_image).png` (URL di modul praktikum 4) |

Link `/files/<id>` GitBook 404 jika diakses langsung; URL asli diambil dengan grep `*-files.gitbook.io` dari HTML halaman.

## Hasil utama (angka dari eksekusi nyata)

**Praktikum 1 — EDA Titanic**
- Dimensi 891 × 12; missing: Age 177, Cabin 687, Embarked 2 (total 866 = 8,10% sel).
- Age mean 29,699 (std 14,526, hanya dari 714 data); Fare mean 32,204, max 512,329.
- Skewness: Age 0,389; Fare 4,787 (right-skew kuat).
- Outlier Fare (IQR): Q1 7,91 / Q3 31,00 / IQR 23,09 / batas [-26,72; 65,63] → 116 baris (13,02%).
- Korelasi terhadap Survived: Fare +0,257, Parch +0,082, Age -0,077, Pclass -0,338.
- Korelasi Age–Fare 0,096 → tidak ada hubungan kuat (sesuai modul).

**Praktikum 2 — Imputasi**
- Age diisi mean 29,6991 (177 baris), Cabin → "DECK" (687 baris), Embarked → modus "S" (2 baris).
- Missing setelah imputasi 0; hasil Age & Cabin identik dengan `Titanic-Dataset-fixed.csv` modul.

**Praktikum 3 — Encoding + scaling**
- Seleksi fitur 12 → 5 kolom; LabelEncoder: Sex {female:0, male:1}, Cabin 148 kategori (DECK = 115).
- StandardScaler Age: mean 29,6991, std populasi 12,9947 → mean 0,0000, std 1,0000, min -2,2532, max 3,8709.

**Praktikum 4 — Citra**
- Asli 512×512×3 uint8 (mean 128,2284) → resize 128×128×3 (49.152 piksel, mean 128,2279) → grayscale 128×128 (mean 124,05, std 46,46).

**Tugas — WBC**
- Mentah 569 × 33 → buang `id` (569 nilai unik) dan `Unnamed: 32` (kosong) → 569 × 31.
- diagnosis B 357 / M 212 (37,26% ganas), encoding B=0, M=1.
- 30 fitur numerik distandardisasi: mean ≈ 0 (maks |mean| 8,37e-16), std populasi = 1,0000.

## Deviasi dari modul (dan sebabnya)

1. **`df['Embarked'].fillna(value=df['Embarked'].mode, inplace=True)`** di modul salah: `.mode` adalah *method*, bukan nilai. Efeknya 2 baris Embarked tetap NaN — terbukti pada `Titanic-Dataset-fixed.csv` bawaan modul yang masih punya 2 NaN. Script memakai `df['Embarked'].mode()[0]` = "S" sehingga Embarked bersih 100%.
2. **Modul menyebut Age "data nominal"** — keliru, Age numerik kontinu; alasan pemakaian mean tetap dipertahankan (distribusi cukup simetris, skew 0,389).
3. **`plt.show()` diganti `savefig`** agar script non-interaktif (backend Agg).
4. **Praktikum 3 membaca `Titanic-Dataset-fixed.csv`** (bawaan modul) sesuai jobsheet, bukan output praktikum 2 — hasil imputasi praktikum 2 disimpan terpisah di `praktikum-2/output/`.
5. **`df.info()`/`describe()` versi teks** menggantikan tampilan notebook; ringkasan juga ditulis ke CSV di `output/`.
6. Label encoding `Cabin` (148 nilai unik) menghasilkan angka ordinal tanpa makna — dipertahankan sesuai modul, tetapi secara teknis one-hot/segmentasi dek lebih tepat.
7. Std setelah scaling terbaca 1,0009 bila memakai `pandas.std()` (ddof=1) karena scaler memakai ddof=0; pengecekan memakai ddof=0 sehingga tepat 1.
