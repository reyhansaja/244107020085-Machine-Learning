# JS15 — Pipeline Machine Learning and Deployment

Jobsheet **JS15** mata kuliah Pembelajaran Mesin, JTI Polinema 2025/2026
(<https://polinema.gitbook.io/jti-modul-praktikum-pembelajaran-mesin-2025-2026/js15-pipeline-machine-learning-and-deployment/praktikum-1>).

Studi kasus: klasifikasi citra **Day vs Night** dengan ekstraksi fitur **HOG** +
**Neural Network**, lalu diekspor sebagai artefak (`model.h5`, `scaler.pkl`) dan
di-deploy sebagai web service **Flask** di dalam container **Docker** ke
**Hugging Face Spaces**.

## Struktur

```
js15-pipeline-machine-learning-and-deployment/
├── data/images/{training,test}/{day,night}   # 240 train + 160 test (salinan dataset JS11)
├── praktikum-1/praktikum1_day_night_hog.ipynb        # training + evaluasi + ekspor artefak
├── praktikum-1/output/                               # grafik + metrics.csv (hasil eksekusi nyata)
├── praktikum-2/praktikum2_deployment_flask_docker.ipynb  # app.py, Dockerfile, uji endpoint
├── praktikum-2/output/                               # prediksi_app.csv, ringkasan_praktikum2.json
├── tugas/tugas_deployment_digit_classifier.ipynb     # pipeline yang sama untuk model lain
├── tugas/output/                                     # grafik, prediksi_tugas.csv, samples/*.png
├── deploy/                                           # ARTEFAK DEPLOYMENT DAY/NIGHT
│   ├── app.py
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── day_night_model.h5   (~52 MB)
│   └── scaler.pkl           (~811 KB)
└── tugas/deploy/                                     # ARTEFAK DEPLOYMENT DIGIT
    ├── app.py, requirements.txt, Dockerfile
    ├── digit_model.h5, digit_scaler.pkl
```

## Cara menjalankan ulang

Butuh Python 3.11 (venv `C:/code/pemmes26/.venv`) dengan
`tensorflow==2.15.1`, `numpy==1.26.4`, `scikit-learn`, `scikit-image==0.24.0`,
`opencv-python-headless`, `flask`, `matplotlib`, `pandas`, `nbconvert`.

```bash
# 1) training + ekspor artefak (menulis deploy/day_night_model.h5 & scaler.pkl)
python -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1200 praktikum-1/praktikum1_day_night_hog.ipynb

# 2) membuat & menguji app.py + Dockerfile lewat flask.test_client()
python -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=600 praktikum-2/praktikum2_deployment_flask_docker.ipynb

# 3) tugas: model digit + app.py + Dockerfile + uji endpoint
python -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=600 tugas/tugas_deployment_digit_classifier.ipynb
```

Menjalankan server secara lokal (bukan Colab):

```bash
cd deploy && python app.py     # http://127.0.0.1:7860
```

## Hasil nyata (dari eksekusi notebook, seed = 42)

| Item | Nilai |
|---|---|
| Jumlah data | 240 train (120 day + 120 night), 160 test (80 + 80) |
| Dimensi fitur HOG | **34.596** (`31×31×2×2×9`, citra 256×256, sel 8×8) |
| Model | Dense(128, relu) → Dense(64, relu) → Dense(1, sigmoid) |
| Training | adam, binary_crossentropy, 20 epoch, batch 32 |
| Akurasi test (160 citra) | **0,9500** (loss 0,7108) |
| Precision / recall / F1 (day) | 0,9737 / 0,9250 / 0,9487 |
| Uji endpoint `/predict` (10 citra test) | **10/10 benar** (100%) |
| Tugas digit (sklearn digits) | akurasi test **0,9806**, endpoint **50/50** gambar uji benar |

Grafik dan metrik: `praktikum-1/output/{confusion_matrix,training_history,training_loss,contoh_hog}.png`,
`praktikum-1/output/metrics.csv`, `praktikum-2/output/prediksi_app.csv`,
`tugas/output/{confusion_matrix_digit,training_history_digit}.png`,
`tugas/output/prediksi_tugas.csv`.

## Deviasi dari modul (dan sebabnya)

1. **Google Drive → path lokal.** Modul memakai
   `from google.colab import drive; drive.mount('/content/drive')`, path
   `/content/drive/MyDrive/FolderAnda/images/...`. Di mesin lokal tidak ada Colab/Drive,
   jadi dataset disalin ke `data/images/` dan path dibuat relatif terhadap notebook.
   Struktur folder (`training|test/day|night`) dan isi file identik.
2. **Tanpa push ke Hugging Face.** Langkah 5–7 modul butuh akun + **Access Token**
   pribadi; ini tidak bisa diotomasi dan token tidak boleh diminta/ditebak. Langkah
   `git clone/add/commit/push` ditulis sebagai panduan manual di notebook Praktikum 2,
   dan yang diverifikasi adalah kode + artefak yang sama lewat `flask.test_client()`.
3. **Docker image tidak dibangun.** Base `tensorflow-cpu` ±1 GB dan **Docker daemon
   tidak aktif** di mesin ini (`docker info` gagal: pipe `dockerDesktopLinuxEngine`
   tidak ada). Sebagai gantinya `Dockerfile` divalidasi statis (direktif wajib,
   `EXPOSE 7860`, `CMD ["python","app.py"]`, library sistem OpenCV) — semuanya OK.
4. **Model tugas dilatih ulang.** Artefak `.h5` dari jobsheet sebelumnya tidak tersimpan
   di repo, jadi tugas memakai model **klasifikasi digit 0–9** (dataset `sklearn.datasets.load_digits`,
   1.797 citra 8×8, offline & cepat) yang dilatih langsung di notebook tugas.
5. **Versi paket lokal dipatok** agar kompatibel dengan TF 2.15.1:
   `numpy==1.26.4` (TF 2.15 menolak numpy ≥ 2), `scikit-image==0.24.0`,
   `opencv-python-headless==4.10.0.84` (headless, tanpa GUI).
   Di dalam Docker tetap pakai daftar `requirements.txt` modul (`tensorflow-cpu`).

## Catatan deployment (ringkas)

- `app.py` memuat `day_night_model.h5` + `scaler.pkl` dari working dir → jalankan
  server dari folder `deploy/` (Docker `WORKDIR /app` sudah benar).
- Preprocessing inference **harus identik** dengan training: BGR→RGB, resize 256×256,
  grayscale, HOG (9, (8,8), (2,2), `L2-Hys`), lalu `scaler.transform`.
  Ketidakcocokan urutan ini adalah penyebab umum prediksi acak saat deploy.
- Port wajib **7860** untuk Hugging Face Spaces.
- Dockerfile memakai user non-root `user` (uid 1000) sesuai standar keamanan HF.
