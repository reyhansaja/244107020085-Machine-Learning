# JS04 - Klasterisasi

Sumber: https://polinema.gitbook.io/jti-modul-praktikum-pembelajaran-mesin-2025-2026/js04-klasterisasi

| Jobsheet | Script | Isi |
|---|---|---|
| Praktikum 1 | `praktikum-1/kmeans_iris.ipynb` | K-Means pada Iris, k=2, SSE, elbow k=1..9, crosstab vs spesies asli |
| Praktikum 2 | `praktikum-2/kmeans_konsep.ipynb` | make_blobs, EM manual, efek random seed, over-cluster, moons vs SpectralClustering, digits + t-SNE, kompresi citra 16 warna |
| Praktikum 3 | `praktikum-3/dbscan.ipynb` | DBSCAN eps=0.3 min_samples=10 pada 750 titik make_blobs, 6 metrik, visualisasi core/noise, grid parameter |
| Tugas 1 | `tugas/tugas_kmeans.ipynb` | Mall_Customers: fitur Age + Annual Income + Spending Score, elbow + silhouette, ringkasan cluster |
| Tugas 2 | `tugas/tugas_dbscan.ipynb` | make_moons 1000 sampel noise 0.05, DBSCAN eps=0.2 min_samples=5, evaluasi, eksperimen eps x min_samples |

## Hasil

**Praktikum 1** (SSE k=2 = 152.37, sama seperti modul)
Elbow: k=1 680.82 | k=2 152.37 | k=3 78.95 | k=4 57.44 | k=5 46.54 ...
Siku di k=3 (78.95 turun tajam setelahnya hanya ~20). Crosstab k=3:
setosa 50/50 bersih, versicolor 47, virginica 36 nyasar ke cluster versicolor -> K-Means kesulitan memisah virginica/versicolor.

**Praktikum 2**
Akurasi K-Means pada digits mentah: 0.744; setelah t-SNE 2D: 0.862.
(Angka modul 0.79 / 0.94 dihasilkan versi sklearn lama + random_state default; selisih wajar.)
Kompresi: 273.280 piksel RGB ditekan jadi 16 warna via MiniBatchKMeans.

**Praktikum 3** (identik dengan modul: 3 klaster, 18 noise)
Homogeneity 0.953 | Completeness 0.883 | V-measure 0.917 | ARI 0.952 | AMI 0.916 | Silhouette 0.626
Grid: eps kecil -> klaster pecah jadi puluhan + noise ratusan; eps 0.5 -> semua jadi satu klaster.

**Tugas 1** (Mall Customers)
Silhouette memilih k=6 (0.4284), elbow juga mulai melandai di k=5-6.
Segmentasi yang didapat: pemuda income sedang, pemuda income rendah tapi spending tinggi,
usia menengah income tinggi spending rendah, usia menengah income tinggi spending tinggi, dll.
(Kalau dosen minta k=5, ganti `k_best = 5` di akhir script.)

**Tugas 2** (make_moons)
eps=0.2, min_samples=5 -> 2 klaster, 0 noise, semua metrik 1.000 (data setelah standardisasi sangat bersih).
Grid: eps 0.05 -> noise 186-1000; eps >= 0.3 -> stabil 2 klaster sempurna. Tabel lengkap: `tugas/output/eksperimen.csv`.
