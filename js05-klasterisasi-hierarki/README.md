# JS05 — Klasterisasi Hierarki (HDBSCAN)

Praktikum JTI Polinema 2025/2026, topik HDBSCAN (Hierarchical Density-Based
Spatial Clustering of Applications with Noise) + evaluasi clustering.

## Struktur

```
js05-klasterisasi-hierarki/
├── data/                      digits.csv (1797x65), iris.csv (150x5) — snapshot sklearn
├── praktikum-1/hdbscan_vs_dbscan.ipynb     Langkah 3-6
├── praktikum-2/hyperparameter_hdbscan.ipynb Langkah 7-12
├── tugas/tugas_hdbscan.ipynb               tugas akhir
└── README.md
```
Grafik & metrik: `<praktikum>/output/*.png`, `*.csv`.

## Menjalankan

```bash
C:/code/pemmes26/.venv/Scripts/python.exe praktikum-1/hdbscan_vs_dbscan.ipynb
C:/code/pemmes26/.venv/Scripts/python.exe praktikum-2/hyperparameter_hdbscan.ipynb
C:/code/pemmes26/.venv/Scripts/python.exe tugas/tugas_hdbscan.ipynb
```
Semua self-contained (path via `pathlib` relatif `__file__`), matplotlib backend
`Agg` + `savefig` (tanpa `show`). Sudah dijalankan, exit code 0.

## Beda dari modul (dan sebabnya)

1. **`hdbscan` (PyPI) tidak tersedia** — gagal build di venv ini. Dipakai
   `sklearn.cluster.HDBSCAN` (sklearn 1.7.2, tersedia sejak 1.3). Konsekuensi:
   - default `min_samples=None` → otomatis sama dengan `min_cluster_size`
     (pustaka `hdbscan` default `min_samples=5`); ini yang membuat beberapa baris
     di `p2_metrik.csv` bernilai identik (mis. `min_cluster_size=5` vs
     `min_samples=5`) — hasil nyata, bukan bug: pada dataset ini partisi
     didominasi `min_samples`.
   - Atribut tree bernama privat `_single_linkage_tree_` (structured array).
2. **`dbscan_clustering(cut_distance=...)` tidak ada di sklearn.** Padanan:
   potong single linkage tree HDBSCAN dengan `scipy.cluster.hierarchy.fcluster`
   (`criterion="distance"`), lalu cluster berukuran `< min_samples` dijadikan
   noise (-1) — meniru perilaku `hdbscan.get_clusters(cut_distance=...)`.
3. **Evaluasi** ditambah ARI terhadap label asli (modul hanya silhouette + DBI);
   silhouette/DBI dilewati bila cluster < 2 (hasil cut terlalu kasar).
4. `plot()` modul memakai `proba_map` berbasis indeks; versi ini langsung
   `probabilities[ci]` (identik hasilnya, lebih ringkas).

## Hasil nyata

### Praktikum 1 (`output/p1_metrik.csv`)

| eksperimen | cluster | noise | silhouette | ARI |
|---|---|---|---|---|
| DBSCAN eps=0.3 scale=1 | 3 | 24 | 0.6652 | 0.9242 |
| DBSCAN eps=0.3 scale=0.5 | 1 | 2 | – | 0.0000 |
| DBSCAN eps=0.3 scale=3 | 11 | 320 | 0.0663 | 0.5903 |
| DBSCAN eps=0.9 scale=3 | 3 | 24 | 0.6652 | 0.9242 |
| HDBSCAN scale=1 / 0.5 / 3 | 3 | 40 | 0.6548 | 0.8941 |
| DBSCAN eps=0.7 multiskala | 3 | 23 | 0.6315 | 0.6464 |
| DBSCAN eps=0.3 multiskala | 14 | 206 | 0.2181 | 0.4041 |
| HDBSCAN multiskala | 4 | 32 | 0.5716 | 0.9212 |

Kesimpulan:
- DBSCAN **tidak** scale-invariant: `eps` tetap 0.3, data diskalakan 0.5x →
  semua titik jadi 1 cluster; 3x → pecah 11 cluster + 320 noise. Setelah `eps`
  ikut diskalakan (0.9 untuk 3x) hasil kembali identik (3 cluster, ARI 0.9242).
- HDBSCAN **invariant** terhadap skala: label, jumlah cluster, noise, bahkan
  silhouette identik di skala 1 / 0.5 / 3 (mutual reachability distance ikut
  terskala linier, ekstraksi cluster berbasis kepadatan relatif).
- Multi-skala (std 0.2 / 0.35 / 1.35 / 1.35): DBSCAN satu `eps` tidak bisa
  menangani keduanya (eps=0.7 → 3 cluster, 2 cluster padat tergabung, ARI 0.646;
  eps=0.3 → 14 cluster, yang renggang pecah, ARI 0.404). HDBSCAN menemukan
  keempat cluster (ARI 0.921, noise 32).

### Praktikum 2 (`output/p2_metrik.csv`)

| eksperimen | cluster | noise | silhouette | DBI | ARI |
|---|---|---|---|---|---|
| min_cluster_size=5 | 4 | 32 | 0.5716 | 1.5931 | 0.9212 |
| min_cluster_size=3 | 90 | 254 | 0.0077 | 1.4236 | 0.0295 |
| min_cluster_size=25 | 4 | 90 | 0.5108 | 1.6190 | 0.8234 |
| min_samples=5 (mcs=20) | 4 | 32 | 0.5716 | 1.5931 | 0.9212 |
| min_samples=3 (mcs=20) | 4 | 19 | 0.5789 | 1.5220 | 0.9515 |
| min_samples=25 (mcs=20) | 4 | 90 | 0.5108 | 1.6190 | 0.8234 |
| cut_distance=0.1 | 3 | 529 | 0.0068 | 1.2311 | 0.2444 |
| cut_distance=0.5 | 3 | 99 | 0.5498 | 1.5204 | 0.5541 |
| cut_distance=1.0 | 1 | 15 | – | – | 0.0005 |

Evaluasi HDBSCAN default: **Silhouette 0.5716**, **DBI 1.5931**.

Kesimpulan:
- `min_cluster_size` terlalu kecil (3) → 90 cluster mikro, ARI jatuh ke 0.03;
  terlalu besar (25) → noise naik 32→90, ARI turun ke 0.82. Sweet spot di sini 5.
- `min_samples` kecil (3) → cluster lebih ketat, noise turun 32→19, ARI terbaik
  0.9515; besar (25) → noise 90, ARI 0.8234.
- `cut_distance` kecil memotong terlalu dalam (0.1 → 529 noise, 3 cluster sisa,
  silhouette ~0); 1.0 terlalu dangkal → semua jadi 1 cluster. Cut 0.5 paling
  seimbang (ARI 0.554) tetapi tetap kalah dari `cluster_selection_method='eom'`
  bawaan HDBSCAN (ARI 0.921) — bukti pemilihan cluster adaptif lebih baik
  daripada potongan datar tunggal.

### Tugas (`output/tugas_metrik.csv`)

| dataset | fitur | mcs | cluster | noise | silhouette | DBI | ARI | NMI |
|---|---|---|---|---|---|---|---|---|
| digits | raw | 5 | 21 | 493 (27.4%) | 0.0721 | 1.9427 | 0.4886 | 0.7365 |
| digits | raw | 10 | 12 | 819 (45.6%) | 0.0396 | 1.8838 | 0.2812 | 0.6435 |
| digits | raw | 15 | 11 | 903 (50.2%) | 0.0183 | 1.9712 | 0.2172 | 0.5948 |
| digits | raw | 25 | 7 | 979 (54.5%) | 0.0414 | 2.1651 | 0.1715 | 0.5284 |
| digits | std | 5 | 21 | 839 (46.7%) | −0.0419 | 2.0271 | 0.2122 | 0.5792 |
| digits | std+pca10 | 5 | 22 | 887 (49.4%) | −0.0842 | 1.6882 | 0.2026 | 0.5664 |
| iris | raw | 5 | 2 | 0 | 0.6867 | 0.3828 | 0.5681 | 0.7337 |
| iris | std | 5 | 2 | 2 | 0.4949 | 0.5344 | 0.5638 | 0.7175 |

Analisis kesesuaian dengan label asli:
- **digits** (10 kelas digit): HDBSCAN menghasilkan 21 cluster pada
  `min_cluster_size=5` — jauh lebih banyak dari 10 kelas. NMI 0.7365 tinggi
  tetapi ARI hanya 0.4886. Artinya struktur yang ditemukan **sebagian cocok**:
  tiap cluster punya satu kelas dominan (lihat "modus kelas per cluster"),
  tetapi beberapa digit terpecah menurut gaya penulisan (digit 1 → 3 cluster:
  27, 48, 67 anggota; digit 5 → 3 cluster: 35, 43, 9 anggota, plus 95 masuk
  noise). Noise 493 titik (27%)
  adalah sampel di daerah transisi antar-digit. Ini perilaku wajar untuk metode
  densitas: yang ditemukan adalah *mode* kepadatan, bukan partisi kelas.
- **iris**: hanya 2 cluster (setosa vs gabungan versicolor+virginica), ARI
  0.568, noise 0. Versicolor dan virginica bertumpuk di ruang 4-D sehingga
  kepadatan keduanya menyatu — HDBSCAN tidak memaksa memisahkannya (berbeda
  dengan k=3 pada K-Means). Setosa terpisah sempurna (50/50 pada fitur mentah,
  48/48 setelah standardisasi).
- Standardisasi/PCA tidak memperbaiki ARI digits (0.489 → 0.212/0.203) karena
  menormalkan variansi tiap piksel justru mengaburkan perbedaan magnitudo goresan.

## Catatan teknis

- Tidak ada paket baru dipasang; seluruhnya numpy/pandas/matplotlib/scipy/sklearn.
- `make_blobs` dipakai apa adanya (tanpa dataset unduhan) sesuai modul;
  digits/iris dari `sklearn.datasets`, snapshot CSV di `data/`.
