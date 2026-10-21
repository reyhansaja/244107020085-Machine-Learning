# JS01 - Pengenalan Pembelajaran Mesin

Pengerjaan jobsheet JTI Polinema 2025/2026:
<https://polinema.gitbook.io/jti-modul-praktikum-pembelajaran-mesin-2025-2026/js01-pengenalan-pembelajaran-mesin>

Satu halaman = satu script Python yang benar-benar dijalankan (exit 0), bukan notebook.
Semua path dihitung dari `Path(__file__)`, grafik disimpan ke `output/` (backend `Agg`).

## Peta halaman -> script

| Halaman GitBook | Script | Output |
|---|---|---|
| `/praktikum-1` (Instalasi Perangkat) | `praktikum-1/instalasi_dan_verifikasi.ipynb` | `praktikum-1/output/versi_library.csv`, `01_smoke_test.png` |
| `/tugas-praktikum-1` Bagian 1 & 2 | `tugas/tugas_praktikum_1.ipynb` | `tugas/output/tugas1_library.csv`, `tugas1_jawaban.md`, `02_ecg_rpeak_scipy.png` |
| `/tugas-pendahuluan-js02` (30 data teman) | `tugas/tugas_pendahuluan_js02.ipynb` | `data/tugas_pendahuluan_js02.csv`, `tugas/output/03_ringkasan.csv`, `03_distribusi.png`, `04_hubungan.png` |
| `/konsep-dasar`, `/etika-dan-tantangan` | materi bacaan; dipakai sebagai dasar jawaban Bagian 2 di `tugas1_jawaban.md` | - |

Cara jalan (venv bersama, jangan pakai python 3.14):

```bash
cd c:/code/pemmes26
.venv/Scripts/python.exe js01-pengenalan-pembelajaran-mesin/praktikum-1/instalasi_dan_verifikasi.ipynb
.venv/Scripts/python.exe js01-pengenalan-pembelajaran-mesin/tugas/tugas_praktikum_1.ipynb
.venv/Scripts/python.exe js01-pengenalan-pembelajaran-mesin/tugas/tugas_pendahuluan_js02.ipynb
```

## Hasil nyata

### Praktikum 1 - verifikasi instalasi
Modul aslinya murni langkah klik (Colab: connect instance, `!pip install mne`; lokal:
Anaconda + env variable `conda`, vscode + kernel). Yang bisa diotomasi adalah
verifikasinya, jadi script mengecek versi python, 10 library wajib, lalu uji asap
melatih model.

- python 3.10.6 (MSC v.1932 64 bit), `cpu count` = 8, platform Windows-10-10.0.26200
- library wajib tersedia **10/10**: numpy 2.2.6, pandas 2.3.3, matplotlib 3.10.9,
  seaborn 0.13.2, scipy 1.15.3, scikit-learn 1.7.2, statsmodels 0.15.0,
  scikit-image 0.25.2, opencv 5.0.0, nltk 3.10.3
- uji asap DecisionTree (`max_depth=3`, split 70/30 stratified, `random_state=42`)
  pada Iris: **akurasi uji 0.9778** (45 data uji, 1 salah)

### Tugas Praktikum 1
Bagian 1 (audit library):

- Terpasang di venv: **1/4** (SciPy 1.15.3). PyPREP, wandb, pyECG tidak terpasang.
- `pip install --dry-run` (dijalankan manual): `pyprep` OK (menarik `mne>=1.3.0`,
  `psutil`, `pooch`), `wandb` OK, **`pyecg` GAGAL** - dependensi
  `ishneholterlib==2017.4.11 -> sklearn` (paket shim lama) harus dibangun dari
  source dan butuh MSVC yang tidak ada di mesin ini.
- Demo SciPy pengganti fungsi inti pyECG: EKG sintetis 10 s @250 Hz, bandpass
  5-25 Hz, `find_peaks` -> **12 denyut, HR 72.0 bpm, galat 0.01 bpm** dari
  ground-truth 72 bpm, RR 833.5 ms (std 1.9 ms).
- Demo `scipy.stats.ttest_ind` (tinggi dua kelompok sintetis): mean 165.10 vs
  170.68 cm, **t = -4.5557, p = 2.735e-05** -> beda signifikan.

Bagian 2 (esai etika + energi/lingkungan) ada di `tugas/output/tugas1_jawaban.md`,
lengkap dengan referensi (Microsoft Responsible AI, Garante/CNIL/ICO Clearview AI,
Autoriteit Persoonsgegevens, IEA *Energy and AI* 2025, Strubell dkk. 2019,
Bender dkk. 2021, Schwartz dkk. *Green AI*).

### Tugas Pendahuluan JS02
30 responden, 7 kolom sesuai modul (+ turunan `imt`), seed 42:

- tinggi: L 170.72 ± 3.87 cm (n=13), P 156.93 ± 3.65 cm (n=17); keseluruhan
  rata-rata 162.90 cm
- berat: L 60.95 ± 5.68 kg, P 50.70 ± 6.51 kg; keseluruhan 55.14 kg
- target IPS: rata-rata **3.444 ± 0.236** (rentang 2.98-4.00), per asal kelas
  TI-1A 3.474 / TI-1B 3.432 / TI-1C 3.392 / TI-1D 3.466
- kota asal terbanyak: Malang 12, lalu Blitar/Kediri/Surabaya/Probolinggo 3
- korelasi: tinggi-berat r = 0.630, berat-IMT r = 0.741, target IPS hampir tidak
  berkorelasi dengan antropometri (|r| < 0.17)

## Deviasi dari modul

1. **Tugas pendahuluan JS02 memakai data simulasi.** Modul minta mengumpulkan data
   30 teman; agen tidak punya akses ke responden nyata, jadi data dibangkitkan
   (`numpy.random.default_rng(42)`, distribusi tinggi/berat wajar per jenis kelamin)
   dengan skema kolom identik. File CSV bisa langsung ditimpa data asli tanpa
   mengubah script. Lihat `data/README.md`.
2. **`pyprep`, `wandb`, `pyecg` tidak dipasang.** Venv dipakai bersama proses lain;
   `mne` (dependensi PyPREP) berisiko menggeser pin numpy/scipy, dan `pyecg` memang
   tidak bisa dibangun tanpa MSVC. Audit status + kegunaan tetap dilaporkan, dan
   fungsi inti pyECG didemonstrasikan dengan `scipy.signal`.
3. **Colab tidak bisa diotomasi** (butuh login Google + instance berbayar). Praktikum 1
   dijalankan sebagai verifikasi pada komputer lokal, dengan tabel padanan perintah
   Colab -> lokal yang dicetak script.
4. **Format spreadsheet = CSV**, bukan `.xlsx`: `openpyxl`/`xlsxwriter` tidak ada di venv
   dan tidak dipasang (aturan tidak menambah paket). CSV terbuka normal di Excel/
   LibreOffice/Google Sheets.
5. Grafik memakai backend `Agg` + `savefig`; tidak ada `plt.show()`.
