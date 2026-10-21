# JS06 — ANN (Approximate Nearest Neighbors)

Praktikum Pembelajaran Mesin JTI Polinema 2025/2026, jobsheet JS06 versi ANN.
Semua script dijalankan nyata di mesin lokal; angka di README berasal dari output
`*/output/run_log.txt` dan `*/output/*.csv`, bukan salinan modul.

## Struktur

```
data/spotify_songs.csv                  # 32.833 track x 9 fitur audio (lihat Deviasi)
praktikum-1/p1_exact_vs_ann.ipynb          # exact vs ANN 2D + simulasi 1jt track 20 fitur
praktikum-2/p2_faiss_ivfpq.ipynb           # FAISS FlatL2 vs IVF+PQ (L2/IP, 1e3/1e6, 2D/5D)
praktikum-3/p3_hnsw.ipynb                  # sklearn brute vs HNSW (L2/IP, 1e3/1e6, 2D/5D)
praktikum-4/p4_bandingkan_tiga.ipynb       # Annoy vs FAISS Flat vs HNSW, 1jt x 5D
praktikum-5/p5_benchmark_recall_qps.ipynb  # kurva recall vs QPS (100k x 128D)
praktikum-6/p6_spotify_ann.ipynb           # ANN pada dataset Spotify
tugas/tugas_benchmark.ipynb                # 10.000 x 128D, 4 metode
```

Jalankan: `C:/code/pemmes26/.venv/Scripts/python.exe praktikum-1/p1_exact_vs_ann.ipynb`
(atau `python <script>` dari root folder ini). Butuh numpy, pandas, scikit-learn,
faiss-cpu, matplotlib.

## Deviasi (jujur)

1. **`annoy` tidak dapat di-build di mesin ini** (butuh MSVC, tidak tersedia). Padanan
   yang dipakai: `faiss.IndexIVFFlat` — sama-sama ANN berbasis partisi/cluster.
   "tree" Annoy dipetakan ke `nlist`, `search_k` ke `nprobe`.
2. **`hnswlib` juga gagal di-build** (alasan sama). Padanan: `faiss.IndexHNSWFlat`
   dengan parameter setara (M=16, efConstruction=100–200, efSearch=50–200).
   Recall/waktu tetap metrik nyata dari faiss, bukan dari hnswlib.
3. Jobsheet praktikum-1 memakai **50 juta track** (4 GB float32). Diperkecil ke
   **1 juta** — angka ini yang diminta tabel jobsheet dan RAM mesin terbatas.
4. Dataset Spotify: link Kaggle butuh login. Dipakai **TidyTuesday "Spotify Songs"**
   (`rfordatascience/tidytuesday`, 32.833 baris) dengan 9 kolom fitur yang identik
   dengan `spotify_songs.csv` Kaggle/TidyTuesday. Ditemukan duplikat track antar
   playlist, jadi tetangga terdekat sering track yang sama.
5. Praktikum-6 menghitung metrik pada 2.000 query pertama (bukan seluruh 32.833
   baris) agar brute force selesai cepat; k=10 tetap.
6. Tugas diminta membandingkan Colab vs PyDroid3. Keduanya tidak tersedia di sesi
   ini, jadi hanya mesin lokal: Windows 11, 8 logical core, Python 3.10.6,
   faiss 1.15.1.
7. Angka waktu exact di praktikum-3 memakai `sklearn NearestNeighbors` (seperti
   jobsheet) sehingga mengandung overhead Python per panggilan; waktu ANN memakai
   faiss, jadi speedup absolut agak bias ke ANN. Perbandingan antar-ANN (p4/p5) adil
   karena semuanya faiss.

## Hasil

### Praktikum 1 — exact vs ANN (k=3, ms/query, 50 query)

| Metrik | tree/nlist | N | ENN index | ANN index | exact ms | ANN ms | build ms | speedup | recall@3 |
|---|---|---|---|---|---|---|---|---|---|
| Euclidean | 3 | 1.000 | [317 381 119] | [317 381 119] | 0,270 | 0,064 | 86 | 4,2x | 0,993 |
| Euclidean | 8 | 1.000 | [317 381 119] | [317 381 119] | 0,142 | 0,262 | 297 | 0,54x | 0,973 |
| Euclidean | 3 | 100.000 | [75453 2977 81227] | [75453 2977 81227] | 4,693 | 0,218 | 270 | 21,5x | 0,987 |
| Angular | 3 | 1.000 | [317 411 38] | [317 411 38] | 0,119 | 0,106 | 127 | 1,1x | 0,987 |
| Angular | 8 | 1.000 | [317 411 38] | [317 411 38] | 0,027 | 0,038 | 153 | 0,73x | 0,980 |
| Angular | 3 | 100.000 | [62844 86475 89113] | [11184 8650 7524] | 3,082 | 0,213 | 275 | 14,5x | 0,160 |

- ANN baru menang saat N besar. Di N=1.000, build + overhead index lebih mahal
  daripada sekadar menghitung semua jarak (speedup < 1).
- Angular 100k recall jatuh ke 0,16 karena `nlist=3` + `nprobe=1`: query hanya
  memeriksa 1 dari 3 partisi (≈33% data) dan sebaran cosine 2D sangat rapat.
- **Kenapa build tidak dihitung waktu?** Build adalah preprocessing offline sekali
  jalan; yang diukur adalah latency query. Di N kecil build (86–300 ms) justru
  jauh lebih lama daripada satu query exact (0,1–5 ms).

Simulasi rekomendasi 1jt track × 20 fitur, k=5, 20 query:
- Exact: 1,13 s total (52,07 ms/query), index `[886500 924476 625732 654741 9149]`
- IVF (build 3,78 s, nlist=100, nprobe=10): 1,374 ms/query → **37,9x** lebih cepat,
  hasil identik untuk top-5, recall@5 = 0,920.

### Praktikum 2 — FAISS Flat vs IVF+PQ (k=3, ms/query, 50 query)

| Metrik | N | d | nlist | flat ms | IVFPQ ms | build ms | speedup | recall@3 |
|---|---|---|---|---|---|---|---|---|
| L2 | 1.000 | 2 | 10 | 0,381 | 0,050 | 346 | 7,5x | 0,907 |
| L2 | 1.000 | 5 | 10 | 0,252 | 0,074 | 1.620 | 3,4x | 0,747 |
| L2 | 1.000.000 | 2 | 100 | 5,716 | 0,129 | 13.701 | 44,3x | 0,807 |
| L2 | 1.000.000 | 5 | 100 | 7,619 | 0,143 | 28.507 | 53,4x | 0,840 |
| IP | 1.000 | 2 | 10 | 0,150 | 0,061 | 689 | 2,5x | 0,380 |
| IP | 1.000 | 5 | 10 | 0,219 | 0,136 | 1.867 | 1,6x | 0,720 |
| IP | 1.000.000 | 2 | 100 | 4,167 | 0,407 | 15.598 | 10,2x | 0,007 |
| IP | 1.000.000 | 5 | 100 | 5,245 | 0,290 | 37.001 | 18,1x | 0,640 |

Analisa: IVFPQ memberi speedup 1,6–53x dengan recall 0,64–0,91. Recall paling
buruk pada IP 2D 1jt (0,007) karena (a) `nprobe=1` hanya membuka 1 dari 100 list,
(b) data 2D ternormalisasi menempati lingkaran satuan sehingga selisih cosine antar
tetangga sangat tipis, dan (c) 8 bit/subvektor kehilangan presisi. Pada 5D hasilnya
wajar (0,64–0,84). Biaya terbesar ada di training/build PQ (13–37 s untuk 1jt).

### Praktikum 3 — sklearn brute vs HNSW faiss (k=5, ms/query, 20 query)

| Metrik | N | d | exact ms | HNSW ms | build ms | speedup | recall@5 |
|---|---|---|---|---|---|---|---|
| L2 | 1.000 | 2 | 41,98 | 0,305 | 282 | 137,8x | 1,000 |
| L2 | 1.000 | 5 | 6,423 | 0,787 | 807 | 8,2x | 1,000 |
| L2 | 1.000.000 | 2 | 17,97 | 0,415 | 66.073 | 43,3x | 1,000 |
| L2 | 1.000.000 | 5 | 11,57 | 0,429 | 79.445 | 27,0x | 1,000 |
| IP | 1.000 | 2 | 0,336 | 0,227 | 73 | 1,5x | 1,000 |
| IP | 1.000 | 5 | 0,216 | 0,183 | 137 | 1,2x | 1,000 |
| IP | 1.000.000 | 2 | 38,10 | 0,389 | 54.576 | 97,9x | 0,060 |
| IP | 1.000.000 | 5 | 50,38 | 0,370 | 65.768 | 136,1x | 1,000 |

HNSW nyaris sempurna untuk L2 (recall 1,000) di semua ukuran. Pengecualian:
IP 1jt 2D recall 0,06 — semua vektor ada di lingkaran satuan, jarak antar tetangga
hampir sama, `efSearch=50` tidak cukup; menaikkan `ef` akan menaikkan recall tapi
memperlambat query. Build HNSW mahal (55–80 s untuk 1jt) dan naik linear.

### Praktikum 4 — 1 juta vektor 5D, k=10, 20 query

| Metrik | Metode | build s | query ms | recall@10 | top-5 |
|---|---|---|---|---|---|
| L2 | FAISS Flat | 0,012 | 9,445 | 1,000 | [85158 586716 687018 8630 91763] |
| L2 | IVF (proxy Annoy) | 1,484 | 0,847 | 1,000 | sama |
| L2 | HNSW (faiss) | 110,464 | 0,274 | 1,000 | sama |
| IP | FAISS Flat | 0,038 | 7,090 | 1,000 | [901365 779224 732589 766000 586977] |
| IP | IVF (proxy Annoy) | 1,644 | 0,669 | 1,000 | sama |
| IP | HNSW (faiss) | 107,557 | 0,094 | 1,000 | sama |

Pada 5D semua ANN mencapai recall 1,0. HNSW tercepat saat query (11–75x flat) tapi
build-nya ~70x lebih lama (110 s vs 1,5 s IVF). Untuk 1jt vektor berdimensi rendah,
IVF adalah kompromi terbaik: build 1,5 s, query < 1 ms.

### Praktikum 5 — 100.000 x 128D, 1.000 query, k=10

| Metode | parameter | recall@10 | QPS |
|---|---|---|---|
| IVF-Flat (proxy Annoy) | nprobe 1/5/10/20 | 0,055 / 0,194 / 0,317 / 0,502 | 20.010 / 4.209 / 2.658 / 1.333 |
| FAISS IVF-PQ | nprobe 1/5/10/20 | 0,044 / 0,102 / 0,133 / 0,167 | 33.005 / 11.850 / 7.291 / 3.620 |
| HNSW (faiss) | ef 50/100/200/400 | 0,305 / 0,450 / 0,614 / 0,766 | 3.853 / 4.047 / 2.074 / 1.123 |
| Exact Flat | — | 1,000 | 521 |

Grafik `praktikum-5/output/p5_recall_vs_qps.png`. Semua metode ANN di atas exact
secara QPS (2–60x) dengan trade-off recall. HNSW mendominasi di recall menengah
(0,45–0,77); IVF-PQ paling cepat tapi recall terendah karena kompresi 8 bit.
Data 128D uniform acak adalah kasus terburuk ANN (curse of dimensionality) —
recall 0,77 pada ef=400 lebih rendah dari data nyata.

### Praktikum 6 — Spotify, 32.833 track x 9 fitur, k=10, 2.000 query

| Metode | build s | ms/query | recall@10 |
|---|---|---|---|
| Exact (sklearn brute) | 0,000 | 0,4026 | 1,000 |
| Annoy → faiss IVF | 0,286 | 0,0278 | 0,989 |
| HNSW → faiss HNSW | 3,066 | 0,1251 | 0,989 |
| FAISS IVF (nprobe=10) | 0,408 | 0,0341 | 0,989 |

Top-5 untuk "I Don't Care (with Justin Bieber) - Loud Luxury Remix" identik di
keempat metode: `[0, 29684, 29734, 4959, 29713]` — track itu sendiri, dua duplikat
rilis yang sama, lalu "I Miss You" (Clean Bandit) dan "Name & Number" (Shift K3Y).
Analisa: 9 fitur hasil StandardScaler punya struktur kuat (lagu pop/EDM mengumpul),
jadi ANN mudah mencapai recall 0,989 dengan query 14x lebih cepat dari brute force.
HNSW kalah cepat di sini karena build graph mahal untuk 33k titik dan `efSearch=200`
membuat query 4x lebih lambat dari IVF; untuk dataset sekecil ini IVF sudah cukup.

### Tugas — 10.000 x 128D, k=10, 10.000 query (detik)

| Metode | build | query | recall@10 |
|---|---|---|---|
| Exact NN (sklearn brute) | 0,0000 | 1,6572 | 1,000 |
| Annoy → faiss IVF | 0,4146 | 0,3562 | 0,414 |
| HNSW → faiss HNSW | 2,4321 | 2,5751 | 0,986 |
| FAISS IVF | 0,2976 | 0,3908 | 0,414 |

Speedup query vs exact: IVF 4,7x, HNSW 0,6x. HNSW lebih lambat dari brute force
pada 10k x 128D karena biaya traversal graph + `efSearch=200` per query melebihi
biaya menghitung 10k jarak, walau recall-nya 0,986. IVF cepat tapi recall hanya
0,414 (`nprobe=10` dari 100 list). Artinya pemilihan ANN harus melihat N: untuk
dataset kecil, brute force masih kompetitif.

Mesin: Windows 11 (`platform.release()` melaporkan "10"), 8 logical core,
Python 3.10.6, faiss 1.15.1. Perbandingan Colab vs PyDroid3 tidak bisa dijalankan
di sesi ini (tidak ada akses ke Colab maupun Android).
