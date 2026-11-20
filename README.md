# Praktikum Pembelajaran Mesin 2025/2026

Pengerjaan jobsheet JTI Polinema
(https://polinema.gitbook.io/jti-modul-praktikum-pembelajaran-mesin-2025-2026).
Satu jobsheet = satu folder, satu praktikum = satu notebook `.ipynb` yang sudah dieksekusi.

## Isi

| Folder | Jobsheet |
|---|---|
| `js01-pengenalan-pembelajaran-mesin/` | Pengenalan Pembelajaran Mesin |
| `js02-pemahaman-data-dan-pra-pengolahan-data/` | Pemahaman Data & Pra Pengolahan |
| `js03-ektraksi-fitur/` | Ekstraksi Fitur |
| `js04-klasterisasi/` | Klasterisasi (K-Means, DBSCAN) |
| `js05-klasterisasi-hierarki/` | Klasterisasi Hierarki (HDBSCAN) |
| `js06-regresi/` | Model Regresi |
| `js06-ann-approximate-nearest-neighbors/` | Approximate Nearest Neighbors (js06 di GitBook dipakai dua kali) |
| `js09-klasifikasi/` | Klasifikasi (KNN, Naive Bayes) |
| `js11-klasifikasi-2/` | Klasifikasi 2 (SVM) |
| `js13-artificial-neural-network-.../` | ANN & Evaluasi Classifier |
| `js14-convolutional-neural-network/` | CNN |
| `js15-pipeline-machine-learning-and-deployment/` | Pipeline ML & Deployment |

JS07, JS08, JS10, JS12 tidak ada di GitBook (penomoran melompat).

Struktur tiap folder: `data/`, `praktikum-N/<nama>.ipynb`, `tugas/`, `<praktikum>/output/`, `README.md`.

## Cara pakai

```bash
cd c:/code/pemmes26
.venv/Scripts/python.exe -m jupyter lab        # buka GUI, Run All
./run.sh                                       # atau eksekusi semua notebook (bash)
run.bat                                        # sama, dari cmd
```

Eksekusi satu notebook:

```bash
.venv/Scripts/python.exe -m nbconvert --to notebook --execute --inplace \
    js04-klasterisasi/praktikum-1/kmeans_iris.ipynb
```

## Lingkungan

- Python 3.10 (`C:\laragon\bin\python\python-3.10`), venv di `.venv/`, daftar paket di `requirements.txt`.
- TensorFlow 2.15.1 + Keras 2.15 dipakai JS13–JS15 (Keras 2 supaya `ImageDataGenerator` masih jalan).
- `annoy` dan `hnswlib` tidak bisa di-build di mesin ini (tidak ada MSVC) → JS06-ANN memakai `faiss`.
- Notebooks memakai shim `__file__` di sel pertama supaya path relatif ke `data/` tetap benar walau tidak dijalankan sebagai script.
