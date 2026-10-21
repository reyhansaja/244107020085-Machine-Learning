# JS03 — Ekstraksi Fitur (JTI Polinema 2025/2026)

Implementasi Python dari jobsheet JS03 "Ekstraksi Fitur": pra-pengolahan data
terstruktur (Titanic), seleksi fitur, splitting data, ekstraksi fitur citra,
dan tugas Wisconsin Breast Cancer.

## Struktur

```
data/                 dataset mentah + hasil turunan
praktikum-1/          ekstraksi_fitur_terstruktur.ipynb  (+ output/*.png)
praktikum-2/          seleksi_fitur.ipynb                (+ output/*.png)
praktikum-3/          splitting_data.ipynb               (+ output/*.png)
praktikum-4/          ekstraksi_fitur_citra.ipynb        (+ output/*.png|*.csv)
tugas/                wbc_preprocessing.ipynb            (+ output/*.png|*.csv)
```

## Cara menjalankan

Interpreter: `C:/code/pemmes26/.venv/Scripts/python.exe` (pandas, sklearn,
matplotlib, seaborn, pillow, scikit-image).

```
cd c:/code/pemmes26/js03-ektraksi-fitur
python praktikum-1/ekstraksi_fitur_terstruktur.ipynb
python praktikum-2/seleksi_fitur.ipynb     # butuh data/Titanic-Dataset-fixed.csv dari p1
python praktikum-3/splitting_data.ipynb    # butuh data/Titanic-Dataset-selected.csv dari p2
python praktikum-4/ekstraksi_fitur_citra.ipynb
python tugas/wbc_preprocessing.ipynb
```

Semua script `self-contained`: path diturunkan dari `__file__`, matplotlib pakai
backend `Agg` + `savefig`, tidak ada `show()`. Jalankan berurutan karena
praktikum 1 menulis `data/Titanic-Dataset-fixed.csv`, praktikum 2 menulis
`data/Titanic-Dataset-selected.csv`.

## Sumber data

- `data/Titanic-Dataset.csv` — lampiran jobsheet praktikum 1 (891×12).
- `data/Titanic-Dataset-fixed.csv` — dibuat ulang oleh praktikum 1 (hasil imputasi).
- `data/Titanic-Dataset-selected.csv` — dibuat ulang oleh praktikum 2 (5 fitur terpilih).
- `data/wbc.csv` — lampiran tugas (569×33).
- `data/Lenna_(test_image).png` — Wikimedia, 512×512 RGB.

## Hasil utama (nilai nyata dari eksekusi)

**Praktikum 1** — missing: Age 177, Cabin 687, Embarked 2. Imputasi: Age mean
29.6991, Cabin "DECK", Embarked modus "S". Fitur baru: FamilySize (1–11, rata
1.905), Title {Mr 517, Miss 185, Mrs 126, Master 40, Rare 23}, AgeBin
{YoungAdult 535, Adult 195, Teen 70, Child 69, Senior 22}, CabinDeck {D 720,
C 59, B 47, E 32, A 15, F 13, G 4, T 1}. Survival rate: Mrs 0.794, Miss 0.703,
Mr 0.157; deck E 0.750 vs D 0.321.

**Praktikum 2** — encoding Sex {male 1, female 0}, Cabin → deck → LabelEncoder
{A 0, B 1, C 2, D 3, E 4, F 5, G 6, T 7, U 8}; imputasi Age median 28.0;
StandardScaler Age → mean 0.000000, std 1.000562. Seleksi 5 kolom → (891, 5).
Korelasi terhadap Survived: Sex −0.543, Pclass −0.339, Cabin −0.100, Age −0.070.
Age baris pertama = −0.592481, sama dengan dataset bawaan modul.

**Praktikum 3** — random split: 712/89/90 (label 439-273, 53-36, 57-33).
Stratified: 712/89/90 (439-273, 55-34, 55-35), proporsi test 0.611/0.389 vs
populasi 0.616/0.384. KFold(4): fold 668/223, 668/223, 668/223, 669/222.
Split 80:20 + KFold(4): train 712, test 179, fold fit 534 / val 178.

**Praktikum 4** — Lenna RGB 512×512, `len(r.histogram()) = 256`, histogram R
identik dengan modul; mean bin R/G/B = 180.22/99.05/105.41. GLCM (8 level,
rata 4 arah): contrast 0.2403, dissimilarity 0.2013, homogeneity 0.9032,
energy 0.3540, correlation 0.9501. HOG 1764 dimensi (orientasi 9, sel 16×16,
blok 2×2). Total vektor fitur 2549 kolom → `output/fitur_citra.csv`.

**Tugas (WBC)** — 569 baris; `id` (569 unik) dan `Unnamed: 32` (0 non-null)
dibuang → 30 fitur. Encoding diagnosis M=1/B=0 (357 B, 212 M). Standarisasi
30 kolom: max|mean| = 8.37e-16, max|std−1| = 2.22e-16. Stratified 80:20 →
train 455 (285 B / 170 M, proporsi M 0.3736), test 114 (72 B / 42 M, 0.3684),
populasi 0.3726. Fitur paling berkorelasi: concave points_worst 0.7936,
perimeter_worst 0.7829, concave points_mean 0.7766.

## Deviasi dari modul

1. **P1 imputasi Embarked** — modul menulis `df['Embarked'].fillna(df['Embarked'].mode)`
   tanpa `()`, sehingga yang disisipkan objek Series (tidak valid). Dipakai
   `mode()[0]` → "S".
2. **P1 Title** — nama "Rothes, the Countess. of…" menghasilkan nilai
   `the Countess` di modul. Prefix `the ` dibuang lebih dulu sehingga masuk
   kategori `Rare`.
3. **P2 Cabin** — modul mengimpor `LabelEncoder` tetapi tidak memakainya, padahal
   `Titanic-Dataset-selected.csv` bawaan berisi Cabin numerik. Cabin
   di-label-encode (U=8, D=3) agar konsisten; hasil Age/standardisasi tetap
   identik dengan dataset bawaan (baris 0 = −0.592481).
4. **P3 pencetakan index** — modul mencetak seluruh index tiap fold. Di sini
   diringkas jadi rentang (`[223-890]`) agar log terbaca; jumlah dan isi split
   tidak berubah (712/89/90 dst. sama persis).
5. **P3 cross validation 2** — modul memanggil `kf2.split(df_train)` padahal
   variabel yang ada `df4_train` (NameError). Dipakai `C_tr` = `df4_train`.
6. **P3 tambahan** — StratifiedKFold(4) pada data latih 80% sebagai pembanding
   (tidak ada di modul): val tiap fold 110/68, 110/68, 110/68, 109/69.
7. **P4 tambahan** — selain histogram 256 bin per kanal (sesuai modul), ditambah
   statistik kanal, GLCM 5 properti, dan HOG 1764 dimensi sebagai contoh fitur
   citra lain; semuanya disimpan ke `output/fitur_citra.csv`.
8. **Tugas** — modul tidak menyebut kolom yang dibuang; di sini `id` dan
   `Unnamed: 32` (seluruhnya kosong) dibuang eksplisit. Split memakai
   `random_state=42` dan `stratify=y`.

Catatan lingkungan: tidak ada dataset teks pada jobsheet ini sehingga BoW/TF-IDF
tidak dipakai; seluruh praktikum berjalan dengan paket yang sudah terpasang di venv.
