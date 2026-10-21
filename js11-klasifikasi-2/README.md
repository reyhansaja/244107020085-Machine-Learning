# JS11 — Klasifikasi 2 (SVM, PCA, GridSearchCV)

Praktikum JTI Polinema 2025/2026, jobsheet **js11-klasifikasi-2**. Semua script
self-contained, path relatif ke `__file__`, matplotlib backend `Agg` (tidak
`show()`, grafik disimpan ke `output/`).

Sumber materi: `https://polinema.gitbook.io/jti-modul-praktikum-pembelajaran-mesin-2025-2026/js11-klasifikasi-2/*`
(salinan markdown mentah ada di `data/js-*.md`).

## Struktur

```
js11-klasifikasi-2/
  data/
    js-praktikum-1..5.md, js-tugas-praktikum.md   # salinan materi
    voice.csv                                     # dataset tugas 1 (3168 x 21)
    images/{training,test}/{day,night}/*.jpg      # dataset praktikum 5 & tugas 2 (400 citra)
  praktikum-1/svm_linier.ipynb         -> output/01..04*.png
  praktikum-2/svm_nonlinier.ipynb      -> output/01..03*.png
  praktikum-3/svm_overlap.ipynb        -> output/01..02*.png
  praktikum-4/wajah_pca_svm.ipynb      -> output/01..03*.png
  praktikum-5/citra_siang_malam.ipynb  -> output/01..03*.png, fitur_{train,test}.csv
  tugas/svm_voice.ipynb                -> output/01_akurasi_voice.png, tabel_performa_voice.csv
  tugas/citra_histogram.ipynb          -> output/01_grid_histogram.png
  <praktikum>/output/run*.log       # log eksekusi nyata
```

Jalankan:

```bash
C:/code/pemmes26/.venv/Scripts/python.exe praktikum-1/svm_linier.ipynb   # dst.
```

## Dataset

| Dipakai | Dataset | Sumber |
|---|---|---|
| praktikum 1–3 | `make_blobs`, `make_circles` | `sklearn.datasets` (sintetis, tanpa unduh) |
| praktikum 4 | LFW `min_faces_per_person=60` (1348 citra, 62x47, 8 kelas) | `fetch_lfw_people` (unduh sekali, cache `~/scikit_learn_data`) |
| praktikum 5, tugas 2 | 400 citra siang/malam (240 train, 160 test) | `images.zip` dari jobsheet (token gitbook) |
| tugas 1 | `voice.csv` (3168 baris, 20 fitur, label male/female seimbang) | file jobsheet (token gitbook) |

## Hasil (output nyata)

**Praktikum 1 — SVM linier data dummy.** 50 data, 2 kelas. SVM `kernel=linear, C=1e10`
akurasi latih **1.0000**, hanya **3 support vector** (sama dengan nilai
`support_vectors_` di modul). N=60 vs N=120 → sama-sama 3 SV, garis keputusan
praktis identik (data jauh dari margin tidak mengubah fit).

**Praktikum 2 — non-linier.** `make_circles(100, factor=0.1, noise=0.1)`.
Kernel linear akurasi latih **0.6800** (gagal). Kernel RBF `C=1e6` → **1.0000**
dengan 10 SV. Sapuan gamma: gamma 0.1/1/10/100 semua akurasi 1.0, jumlah SV naik
6 → 9 → 40 → 74 (gamma besar = batas makin kompleks/overfit).

**Praktikum 3 — data overlapping.** Efek C (kernel linear, `cluster_std=0.8`):

| C | akurasi latih | #SV | margin 2/‖w‖ |
|---|---|---|---|
| 0.01 | 0.9900 | 52 | 3.5802 |
| 0.1 | 1.0000 | 20 | 2.0380 |
| 1 | 1.0000 | 9 | 0.8740 |
| 10 | 1.0000 | 4 | 0.4702 |
| 100 / 1e4 | 1.0000 | 3 | 0.4030 |

C kecil → margin lebar + banyak SV (margin lunak); C besar → margin sempit.

**Praktikum 4 — wajah LFW, PCA(150)+SVC(RBF).** Grid 4x4 (C × gamma), cv=3:
`best_params = {'svc__C': 5, 'svc__gamma': 0.001}`, `best_score = 0.8170`.
Akurasi test **0.8576** (48/337 salah), macro-F1 **0.8110**, weighted-F1 0.8559.
Akurasi per kelas: Powell 0.9153, Schroeder 0.9259, Bush 0.9023, Blair 0.8611,
Rumsfeld 0.8333, Koizumi 0.7333, Sharon 0.6842, Chavez 0.5556.

**Praktikum 5 — siang/malam, fitur rata-rata kecerahan (kanal V HSV).**
V rata-rata: day 137.44, night 69.28.
Threshold 120 → akurasi train **0.8417**, test **0.8688** (21/160 salah) —
sama dengan modul. SVM RBF default → train **0.8625**, test **0.9000**; report
test: night P 0.8810 / R 0.9250 / F1 0.9024, day P 0.9211 / R 0.8750 / F1 0.8974,
confusion matrix `[[74, 6], [10, 70]]`. Sapuan threshold: terbaik di 100
(train 0.9042 / test 0.9250), threshold 120 bukan yang optimal.

**Tugas 1 — voice.csv, 2 split × 3 kernel (fitur distandarisasi).**

| kernel | 70:30 | 80:20 |
|---|---|---|
| linear | 0.9790 | 0.9748 |
| poly (deg 3) | 0.9600 | 0.9558 |
| rbf (C=10) | 0.9821 | **0.9890** |

Terbaik: RBF + split 80:20, akurasi **0.9890**, F1 0.9889 (female P 0.9844 /
R 0.9937; male P 0.9936 / R 0.9842). Kernel polynomial paling lemah di semua split.

**Tugas 2 — histogram HSV (H16+S8+V8 = 32 dim), SVM RBF, split 80:20.**
Grid gamma × C: akurasi test 0.9125–**1.0000**. Terbaik gamma=0.05, C=1 →
train 1.0000, test 1.0000 (night/day P=R=F1=1.0000, CM `[[40,0],[0,40]]`).
Cek kebocoran data: dari 400 file hanya 1 citra identik (md5) muncul di train
dan test (`20151102_224922.jpg`), jadi hasil ini bukan karena duplikasi —
siang vs malam memang sangat terpisah pada histogram HSV.

## Deviasi dari modul dan sebabnya

1. **Praktikum 2** — `ipywidgets.interact` diganti satu plot 3D statis
   (`elev=30, azim=30`) karena tidak ada kernel interaktif di script; isi
   visual sama. `make_circles` diberi `random_state=0` (modul tanpa seed) agar
   angka reproducible.
2. **Praktikum 4** — GridSearchCV `cv=3` (modul default 5) + `n_jobs=-1`, agar
   selesai < 2 menit; 16 kombinasi selesai 18 detik. `train_test_split` memakai
   `stratify` (modul tidak), sehingga jumlah support per kelas berbeda
   (mis. Sharon 19 vs 15) dan akurasi test 0.8576 vs 0.85 di modul; parameter
   terbaik tetap sama (C=5, gamma=0.001).
3. **Praktikum 5** — fitur rata-rata kecerahan dihitung dari citra asli, bukan
   dari hasil `cv2.resize(1100,600)`; rata-rata kanal V invarian terhadap skala
   (resize hanya dipakai untuk demo shape/visualisasi) dan memotong waktu jalan
   dari 110 s ke 37 s. Akibatnya akurasi train SVM bergeser tipis 0.8583 → 0.8625;
   akurasi test tetap 0.9000 dan hasil threshold identik dengan modul.
4. **Tugas 1** — fitur `voice.csv` distandarisasi (`StandardScaler` dalam
   pipeline) karena SVM sensitif skala; jobsheet tidak menyebut langkah ini.
   Split memakai `stratify=y, random_state=42`.
5. **Tugas 2** — fitur histogram (bukan rata-rata kecerahan), 32 dimensi,
   hyperparameter tuning eksplisit; jobsheet hanya meminta "fitur histogram".
6. **Praktikum 3** — ditambah tabel kuantitatif (akurasi, #SV, margin 2/‖w‖)
   selain gambar; modul hanya menampilkan gambar C=10 vs C=0.1.

## Catatan lingkungan

Python `C:/code/pemmes26/.venv/Scripts/python.exe` (sklearn 1.7.2, numpy 2.2.6,
pandas 2.3.3, cv2 5.0.0, matplotlib 3.10.9). Waktu jalan: p1–p3 ≤ 8 s,
p4 ~34 s (setelah cache LFW), p5 ~23 s, tugas 1 ~7 s, tugas 2 ~7 s.
