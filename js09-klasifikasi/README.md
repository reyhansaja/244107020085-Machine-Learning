# JS09 — Klasifikasi (kNN & Naive Bayes)

Praktikum Pembelajaran Mesin, JTI Polinema 2025/2026.
Sumber materi: https://polinema.gitbook.io/jti-modul-praktikum-pembelajaran-mesin-2025-2026/js09-klasifikasi

## Struktur

```
js09-klasifikasi/
├── data/                 iris.csv, spam.csv, voice.csv  (unduh dari GitBook modul)
├── praktikum-1/knn_iris.ipynb            kNN pada Iris
├── praktikum-2/naive_bayes_dummy.ipynb   MultinomialNB vs GaussianNB, data dummy
├── praktikum-3/naive_bayes_spam.ipynb    MultinomialNB + BoW pada spam.csv
├── tugas-1/knn_voice.ipynb               kNN male/female + seleksi fitur & k
├── tugas-2/mnb_count_vs_tfidf.ipynb      CountVectorizer vs TF-IDF (+stop_words)
└── <praktikum>/output/                grafik PNG + CSV hasil
```

## Cara menjalankan

```bash
C:/code/pemmes26/.venv/Scripts/python.exe praktikum-1/knn_iris.ipynb
# ... dst untuk tiap folder
```

Semua script self-contained: path data dihitung dari `__file__` (`../data`),
matplotlib memakai backend `Agg` dan menyimpan grafik ke `output/`.

## Dataset

| File | Asal | Catatan |
|---|---|---|
| `iris.csv` | GitBook modul `/files/q5xT8JXVVXl11pHjTaQb` | 150 baris, 4 fitur, 3 kelas |
| `spam.csv` | GitBook modul `/files/Y3JrF6vgkudjzIIEKu57` | 5572 SMS, kolom `v1` (label) & `v2` (teks) + 3 kolom kosong |
| `voice.csv` | GitBook modul `/files/cIcgnsUamQwJzOUaDUMX` | 3168 baris, 20 fitur akustik, label male/female |

## Hasil

**Praktikum 1 — kNN Iris** (split 70/30, `random_state=42`, standardisasi)
- k=3: akurasi **1.0000**, confusion matrix `[[19,0,0],[0,13,0],[0,0,13]]`, precision/recall/F1 = 1.00 semua kelas.
- Sweep k=1..10: akurasi uji 0.9778 (k=1,2,4) lalu 1.0000 dari k=3,5..10; akurasi train turun 1.0000 → 0.9333 (k=10). Grafik: `k_vs_akurasi.png`.

**Praktikum 2 — Naive Bayes data dummy** (30 sampel, 2 fitur, `random_state=42`)
- MultinomialNB: train 0.8095, test 0.3333 (CM `[[2,0],[6,1]]`).
- GaussianNB: train 0.8571, test 0.4444 (CM `[[2,0],[5,2]]`).
- Rata-rata 10 seed: MNB 0.5444 ± 0.2191, GNB 0.3778 ± 0.1736.

**Praktikum 3 — MultinomialNB spam** (split 80/20, `random_state=50`, BoW 7727 fitur)
- Akurasi train 0.9946, test **0.9776**, CM `[[951,3],[22,139]]`.
- spam: precision 0.9789, recall 0.8634, F1 0.9175; ham F1 0.9870.

**Tugas 1 — kNN voice.csv** (split 80/20 stratified)
- Seleksi fitur ANOVA F + 5-fold CV: terbaik **6 fitur** `meanfun, IQR, Q25, sp.ent, sd, sfm` → CV 0.9799 (semua 20 fitur hanya 0.9700).
- Sweep k ganjil 1..29: CV terbaik **k=3** (0.9814); akurasi test 0.9874, CM `[[313,4],[4,313]]`, F1 macro 0.9874.

**Tugas 2 — Count vs TF-IDF (stop_words='english')**
| Fitur | acc train | acc test | F1 spam |
|---|---|---|---|
| CountVectorizer | 0.9939 | **0.9821** | 0.9310 |
| TF-IDF | 0.9843 | 0.9731 | 0.8881 |

Kesimpulan: CountVectorizer lebih baik pada kasus ini (+0.9 poin akurasi, F1 spam +4.3 poin);
TF-IDF menekan semua spam-recall (0.7987) meski precision spam 1.0.

## Deviasi dari modul

1. **Praktikum 2** — modul tidak menetapkan `random_state` pada `make_classification`,
   sehingga hasil (0.6190/0.7778 dan 0.6667/0.5556) tidak dapat direproduksi. Di sini seed
   dipatok `random_state=42` (hasil 0.8095/0.3333 dan 0.8571/0.4444) dan variansi seed
   dilaporkan eksplisit. Penyebab beda: n=30 → data uji hanya 9 sampel, satu kesalahan
   menggeser akurasi 11 poin; modul juga menyebut hasil ini "tidak jauh berbeda" padahal
   selisih 22 poin pada data uji.
2. **Praktikum 1/3** — akurasi identik dengan modul (1.0 dan 0.9946/0.9776); ditambah
   confusion matrix & classification_report yang tidak dicetak modul untuk praktikum 3,
   plus contoh SMS yang salah klasifikasi.
3. **Tugas 1** — modul tidak menentukan metode seleksi fitur; dipakai ANOVA F + uji
   kumulatif dengan CV 5-fold agar pemilihan tidak membocorkan data uji.
4. **Tugas 2** — split memakai `stratify=y` (modul tidak), sehingga angka sedikit berbeda
   dari praktikum 3 (0.9821 vs 0.9776).
