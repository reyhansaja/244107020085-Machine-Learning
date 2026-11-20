# Praktikum 14 — Convolutional Neural Network (CNN)

Notebook sudah **dieksekusi** (output + grafik inline), angka di bawah semuanya hasil eksekusi nyata.

```
js14-convolutional-neural-network/
├── data/
│   ├── cifar10_subset.npz          # 20.000 citra latih + 1.967 citra uji (32x32, 10 kelas) + label
│   ├── cat_dog/{training,test}/{cat,dog}/*.png   # 1000 latih + 200 uji per kelas, 64x64
│   └── single_prediction/cat_or_dog_1.jpg
├── praktikum-1/conv_cat_dog.ipynb   + output/
├── praktikum-2/conv_cifar10.ipynb   + output/
├── tugas/tugas_cnn.ipynb            + output/
└── README.md
```

## Hasil (angka nyata dari eksekusi)

| Notebook | Isi | Hasil |
|---|---|---|
| praktikum-1 | Cat vs Dog, `ImageDataGenerator.flow_from_directory`, 2 Conv + 2 Pool + Dense, prediksi 1 citra | **test_acc = 0.6641** (387 citra uji); prediksi citra tunggal benar (`cat`, p(dog)=0.21) |
| praktikum-2 | CIFAR-10, 3 Conv, plot 25 sampel, `model.summary()`, grafik akurasi | **test_acc = 0.6274**, test_loss 1.0818 (10 epoch, 20.000 latih — sesuai `epochs=10` modul) |
| tugas bagian 1 | Perbandingan 4 varian CNN | V1 0.6396 · V2 0.6202 · V3 0.5963 · **V4 (transfer learning) 0.8312 → target >80% tercapai** |
| tugas bagian 2 | CNN MNIST | **akurasi uji = 0.9879**, 121 dari 10.000 salah klasifikasi (contoh ditampilkan) |

Tabel varian juga tersimpan di `tugas/output/perbandingan_varian.csv`:

| varian | epoch | val_acc_akhir | test_acc |
|---|---|---|---|
| V1 (baseline 3 Conv) | 4 | 0.6285 | 0.6396 |
| V2 (+Dropout 0.4) | 4 | 0.6125 | 0.6202 |
| V3 (+konv ganda, BatchNorm, augmentasi) | 8 | 0.6090 | 0.5963 |
| V4 (EfficientNetB0 ImageNet, base dibekukan, 128x128) | 20 | 0.8395 | **0.8312** |

## Deviasi dari modul (dan sebabnya)

1. **Dataset Cat vs Dog (praktikum-1).** Modul memakai arsip Google Drive (ratusan MB) yang tidak bisa diunduh otomatis di
   lingkungan ini. Dipakai padanan **kelas cat (3) dan dog (5) dari CIFAR-10 asli**, di-resize ke **64x64** sesuai
   `target_size` modul, 1000 citra latih + 200 citra uji per kelas. Konsekuensi: citra aslinya 32x32 (dinaikkan ke 64x64)
   dan objeknya kecil/blur, jadi akurasi wajar lebih rendah daripada dataset Cat/Dog resolusi tinggi: **0.6641**.
   Generator uji terisi 387 citra (13 citra gagal terunduh saat pengambilan data).
2. **CIFAR-10 (praktikum-2 & tugas).** `keras.datasets.cifar10.load_data()` mengunduh ~170 MB dari
   `cs.toronto.edu`; di mesin ini unduhan putus (koneksi reset, ±60 KB/detik) dan file tar.gz yang ada korup. Sebagai
   gantinya, subset **CIFAR-10 asli** ditarik dari mirror Hugging Face `uoft-cs/cifar10` lalu disimpan sebagai
   `data/cifar10_subset.npz`: **20.000 latih + 1.967 uji**, resolusi asli 32x32, 10 kelas. Jadi notebook memakai
   `np.load`, bukan `load_data()`; arsitektur, normalisasi 1/255, `to_categorical`, dan alur pelatihan tetap seperti modul.
3. **Epoch / waktu (CPU-only, tanpa GPU).** Batas praktis ±10 menit per notebook membuat epoch dibatasi:
   praktikum-2 10 epoch (sama dengan modul, `validation_data` = data uji); tugas V1 4, V2 4, V3 8, V4 20 epoch head. **V3 belum konvergen** pada 8 epoch (augmentasi
   memperlambat konvergensi), sehingga angkanya di bawah baseline — ini hasil apa adanya, bukan bug.
4. **Cara mencapai >80% (tugas).** CNN from-scratch pada 20.000 citra butuh puluhan epoch/GPU untuk melewati 80%
   (V1–V3 berhenti di 0.60–0.64 dalam batas waktu). Karena itu V4 memakai **transfer learning EfficientNetB0**
   pra-latih ImageNet (`weights="imagenet"`, base dibekukan, citra di-resize ke 128x128, hanya head Dense dilatih)
   → **0.8312**. Bobot ImageNet diunduh sekali dari storage.googleapis.com (bukan dataset).
5. **Generator citra.** Modul memakai `from keras.preprocessing.image import ImageDataGenerator`. Di Keras 2.15 masih
   tersedia (deprecated), jadi praktikum-1 tetap memakainya agar `flow_from_directory` identik dengan modul.
   Augmentasi (rotasi 15°, shift 0.1, zoom 0.1, flip) ditambahkan di praktikum-1 untuk menekan overfitting.
6. **MNIST** diambil dari `keras.datasets.mnist` (cache lokal `mnist.npz`): 60.000 latih / 10.000 uji, 3 epoch, batch 128.

## Menjalankan ulang

```bash
C:/code/pemmes26/.venv/Scripts/python.exe -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=2400 praktikum-1/conv_cat_dog.ipynb
```

Seed dipatok (`tf.random.set_seed(42)`, `np.random.seed(42)`), jadi angka bisa sedikit bergeser antar-run karena
non-determinisme thread CPU, tetapi tidak mengubah kesimpulan.
