# JS13 — Artificial Neural Network (ANN) dan Evaluasi Classifier

Jobsheet JTI Polinema 2025/2026, base URL:
`https://polinema.gitbook.io/jti-modul-praktikum-pembelajaran-mesin-2025-2026/js13-artificial-neural-network-ann-dan-evaluasi-classifier/`

Semua deliverable berupa **notebook Jupyter yang sudah dieksekusi** (output + grafik inline, 0 error cell).
Angka di bawah diambil dari output eksekusi nyata, bukan dari modul.

## Halaman -> notebook

| Halaman modul | Notebook | Isi |
|---|---|---|
| `praktikum-1` | `praktikum-1/jst_2layer_xor.ipynb` | JST 2 layer dari nol (NumPy) pada XOR, forward + backprop manual |
| `praktikum-2` | `praktikum-2/keras_iris.ipynb` | Keras pada Iris (10-8-3, adam, categorical_crossentropy) |
| `praktikum-3` | `praktikum-3/keras_regresi.ipynb` | (a) regresi dummy luas->harga, (b) MLP regresi data Boston |
| `tugas-praktikum` | `tugas/mnist_mlp.ipynb` | Klasifikasi MNIST, 2 hidden layer + eksperimen |

Artefak pendukung: `*/output/*.png` (13 gambar), `*/output/*.csv` (tabel hasil), `data/` (kosong —
dataset diunduh lewat pustaka: `load_iris` sklearn, `fetch_openml('boston')`, `keras.datasets.mnist`).

## Cara menjalankan

```bash
cd C:/code/pemmes26
./.venv/Scripts/python.exe -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=2400 js13-*/praktikum-1/jst_2layer_xor.ipynb   # dst.
```

Seed dipatok di semua notebook: `np.random.seed(42)`, `tf.keras.utils.set_random_seed(42)`,
`random_state=42`, dan sweep seed eksplisit `0..4` pada praktikum-1.

## Hasil (output eksekusi)

### Praktikum 1 — JST 2 layer XOR (lr=0.1, 10.000 epoch)

| Konfigurasi | Loss akhir (5 seed) | Akurasi biner | Konvergen (loss<0.01) |
|---|---|---|---|
| hidden=2 sigmoid (modul) | 0.1274 / 0.0035 / 0.0027 / 0.0024 / 0.0030 | 0.90 rata2 | 4 dari 5 seed |
| hidden=3 sigmoid (tugas) | rata2 0.0023 (maks 0.0027) | 1.00 | 5 dari 5 seed |
| hidden=2 relu | rata2 0.1834 | 0.75 | 0 dari 5 seed |
| hidden=3 relu | rata2 0.0585 | 0.90 | 3 dari 5 seed |

* Seed 42 (kebetulan buruk): hidden=2 -> loss 0.128227, prediksi `[0.053, 0.4955, 0.9509, 0.5032]`, akurasi 0.5.
* Hidden=3 seed 42: loss **0.002297**, prediksi `[0.0252, 0.9526, 0.9512, 0.0627]`.
* ReLU hidden=3 seed 42: loss **0.000352** (lebih rendah dari sigmoid).
* Uji arah update: `W += lr*d` (kode modul) vs `W -= lr*grad(MSE)` dengan `lr=0.2` menghasilkan kurva
  **identik** (selisih maksimum 0.00e+00); `W -= lr*d` (tanda dibalik) berhenti di loss 0.499817,
  prediksi semua ~0.5 (gagal belajar).

### Praktikum 2 — Keras Iris (50 epoch, batch 8, adam)

Model dasar 10-8-3: **loss uji 0.207368, akurasi uji 0.9667** (30 sampel uji; versicolor 0.889 recall).

| Hidden layer (relu) | loss uji | akurasi uji |
|---|---|---|
| 4 | 0.3708 | 0.9333 |
| 8 | 0.3626 | 0.9667 |
| 10-8 (modul) | 0.2074 | 0.9667 |
| 16-16 | 0.1754 | 1.0000 |
| 32-16 | 0.0928 | 1.0000 |
| 64-32-16 | 0.0688 | 0.9667 |

Sigmoid vs ReLU (rata-rata 3 arsitektur): loss uji **0.4926 vs 0.2484**, akurasi **0.9222 vs 0.9778**,
akurasi pada epoch ke-10 **0.6556 vs 0.7250** (sigmoid konvergen lebih lambat).

### Praktikum 3 — regresi

(a) Dummy luas->harga (100 epoch, 4 sampel latih). Tugas 4, pengaruh learning rate:

| learning rate | loss awal | loss akhir |
|---|---|---|
| 1e-4 | 1.066133 | 0.988751 |
| 1e-3 (modul) | 1.066133 | 0.600235 |
| 1e-2 | 1.066133 | 0.001676 |
| 1e-1 | 1.066133 | **0.000010** |
| 1.0 | 1.066133 | 0.250079 (tidak stabil) |

Prediksi satu-satunya sampel uji: harga asli 600, prediksi 697.7 (model 10 relu + 1 linear, lr=1e-3).

(b) Boston (OpenML, 506 x 13, split 80/20, 200 epoch, batch 32): **RMSE validasi 3.2354**
(ribu USD), MAE 2.0946; MSE latih 4.9326 vs validasi 10.4680 (overfit ringan).

### Tugas praktikum — MNIST (5 epoch, batch 128, 60.000 latih / 10.000 uji, CPU)

| Eksperimen | akurasi uji | loss uji | waktu latih |
|---|---|---|---|
| 256-128 relu (dasar) | 0.9771 | 0.0752 | 12.0 s |
| 64-32 relu | 0.9666 | 0.1092 | 7.4 s |
| 256-128-64 relu (tambah 1 hidden) | **0.9779** | 0.0739 | 12.3 s |
| 256-128 sigmoid | 0.9639 | 0.1215 | 12.8 s |

Total seluruh eksperimen MNIST < 1 menit (jauh di bawah batas 10 menit).

## Deviasi dari modul + sebab

1. **Praktikum 1 — klaim "bug tanda" tidak terbukti.** `d_W2 = a1^T @ (error * sigmoid'(a2))`
   adalah **negatif** gradien MSE (`dL/dW2 = -0.5 * d_W2`), jadi `W += lr*d` SUDAH gradient descent.
   Dibuktikan: identik sampai 0.00e+00 dengan `W -= lr*grad(MSE)`. Membalik tanda (`W -= lr*d`,
   seperti saran "perbaikan") justru membuat loss macet di 0.4998. Masalah nyata modul:
   (a) tanpa seed -> hasil tidak reproducible, (b) 2 neuron hidden + sigmoid mudah terjebak local
   minimum (seed 0 dan seed 42 berhenti di ~0.128, akurasi 0.5). Perbaikan yang diterapkan:
   seed dipatok, hidden=3, dan pembanding ReLU.
2. **Praktikum 2 — `OneHotEncoder(sparse=False)`** sudah dihapus di scikit-learn 1.2+
   (TypeError di sklearn 1.7.2) -> dipakai `sparse_output=False`.
   `train_test_split` modul tanpa `random_state` -> ditambah `random_state=42` supaya split tetap.
   `tf.keras.__version__` tidak ada di Keras 2.15 (namespace `keras.api._v2`) -> dicetak dari
   `import keras`.
3. **Praktikum 3 — `load_boston` dihapus sejak scikit-learn 1.2** -> dipakai pengganti resmi
   `fetch_openml(name='boston', version=1, as_frame=True)` (id OpenML 531, 506x13, kolom MEDV),
   berhasil tanpa fallback. Jalur fallback `fetch_california_housing` tetap ada di sel `try/except`
   dan akan mencetak nama dataset pengganti bila OpenML gagal.
4. **Praktikum 3 (a)** modul menstandardisasi `y` juga; prediksi dilaporkan dua kali
   (skala ternormalisasi dan hasil `inverse_transform` = rupiah/ribu USD asli).
5. **Tugas** modul tidak menyebut jumlah epoch; dibatasi 5 epoch / batch 128 / 60k-10k sampel
   supaya total < 10 menit, dan waktu latih dicatat per model (`output/hasil_eksperimen_mnist.csv`).
6. **Lingkungan**: TensorFlow 2.15.1 butuh `numpy<2`. Sempat ada `numpy 2.2.6` di venv bersama yang
   membuat `import tensorflow` gagal (`_ARRAY_API not found`); saat eksekusi final venv memakai
   **numpy 1.26.4** dan semua notebook berjalan. Disk `C:` juga sempat 100% penuh sehingga satu
   notebook gagal ditulis; pip cache dibersihkan (1,0 GB) agar ada ruang. Tidak ada
   `pip install/uninstall` ke venv bersama.
