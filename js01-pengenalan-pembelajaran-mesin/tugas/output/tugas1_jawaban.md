# Tugas Praktikum 1 - Jawaban## Bagian 1 - Library

| Library | Status di venv ini | Kegunaan |
|---|---|---|
| PyPREP | TIDAK ADA (ModuleNotFoundError) | Preprocessing EEG otomatis (implementasi PREP): deteksi kanal buruk, filter line-noise, interpolasi kanal rusak, rujukan (reference). Dipakai sebelum analisis/klasifikasi sinyal EEG. |
| SciPy | TERPASANG (v1.15.3) | Komputasi ilmiah: aljabar linear, optimasi, interpolasi, statistika (scipy.stats), pengolahan sinyal (scipy.signal), integrasi. Fondasi banyak library ML. |
| wandb | TIDAK ADA (ModuleNotFoundError) | Weights & Biases: pencatatan eksperimen (metrik per epoch, kurva, artefak model), pembandingan run, dan pelaporan hasil secara daring. |
| pyECG | TIDAK ADA (ModuleNotFoundError) | Pengolahan sinyal EKG: pembacaan format rekaman (WFDB/ISHNE Holter), deteksi gelombang (P-QRS-T), ekstraksi fitur denyut. |

Cara instalasi (sesuai modul, dijalankan di Colab/lokal):

```
pip install pyprep scipy wandb pyecg
```

Catatan eksekusi nyata: `pyprep` dan `wandb` dapat di-resolve oleh pip pada
venv ini (dry-run berhasil, PyPREP menarik `mne>=1.3`), tetapi tidak dipasang
karena venv dipakai bersama proses lain dan penambahan `mne` berisiko
menggeser pin numpy/scipy. `pyecg` GAGAL di-resolve: dependensinya
`ishneholterlib==2017.4.11 -> sklearn` (paket shim lama) yang harus dibangun
dari source dan butuh MSVC (tidak tersedia di mesin ini). Karena itu fungsi
inti pyECG (baca sinyal EKG + deteksi QRS) didemonstrasikan ulang memakai
`scipy.signal` di script ini.## Bagian 2.1 - Contoh pelanggaran etika dan hukum AI

**Kasus: pengumpulan wajah massal tanpa persetujuan (Clearview AI).**
Clearview AI mengambil miliaran foto wajah dari internet dan media sosial
tanpa izin untuk membangun basis data pengenalan wajah yang dijual ke penegak
hukum dan swasta. Otoritas perlindungan data di Italia (Garante), Prancis
(CNIL), dan Inggris (ICO) menjatuhkan denda serta memerintahkan penghapusan
data karena melanggar GDPR/prinsip pemrosesan yang sah, dan sejumlah lembaga
lain menyimpulkan praktik ini melanggar privasi.

Pelanggaran yang terjadi:
1. **Privacy** - data biometrik dipanen tanpa dasar hukum dan tanpa
   persetujuan subjek.
2. **Transparency & accountability** - subjek data tidak tahu wajahnya dipakai
   dan tidak punya mekanisme koreksi.
3. **Fairness** - akurasi pengenalan wajah lebih rendah pada kelompok tertentu
   (kulit gelap, perempuan), sehingga risiko salah tuduh tidak merata.

Contoh lain yang sering dirujuk: skandal tunjangan anak di Belanda
(*toeslagenaffaire*) - sistem klasifikasi risiko yang bias etnis/kelas
menyebabkan ribuan keluarga dituduh curang secara keliru, dan sistem skor
kredit yang menolak pinjaman berdasarkan atribut proksi (kode pos, jenis
perangkat) alih-alih kemampuan bayar.

Referensi:
- Microsoft, *Responsible AI - Principles and Approach*,
  https://www.microsoft.com/en-us/ai/principles-and-approach
- Garante per la protezione dei dati personali (2022), provvedimento su
  Clearview AI, https://www.garanteprivacy.it
- CNIL (2022), sanction Clearview AI, https://www.cnil.fr
- ICO (2022), enforcement notice Clearview AI, https://ico.org.uk
- Autoriteit Persoonsgegevens (2021), boete toeslagenaffaire,
  https://www.autoriteitpersoonsgegevens.nl## Bagian 2.2 - Dampak energi & lingkungan dan cara mengatasinya

Dampak yang nyata:
1. **Fase fabrikasi** - produksi chip akselerator (GPU/TPU/HBM) memakai
   energi, air ultramurni, dan bahan kimia per unit yang jauh lebih besar
   daripada chip biasa; jejak karbonnya "dibayar di muka" sebelum model
   dipakai.
2. **Fase pelatihan** - pra-pelatihan model besar butuh ribuan GPU-jam.
   Strubell dkk. (2019) menunjukkan biaya energi/emisi pelatihan model NLP
   dapat menyamai emisi beberapa kali penerbangan lintas benua, dan biaya itu
   berlipat saat tuning hiperparameter tanpa pelaporan.
3. **Fase inferensi** - biaya terbesar jangka panjang karena berjalan terus
   menerus; layanan GenAI menambah beban listrik dan air pendingin pusat
   data. Laporan IEA *Energy and AI* (2025) memproyeksikan konsumsi listrik
   pusat data melonjak terutama karena akselerator AI.
4. **Efek ikutan** - rebutan air tanah di daerah kering, limbah e-waste
   akselerator, dan naiknya harga listrik lokal di sekitar pusat data.

Cara mengatasi (pendapat pribadi, dari yang termurah):
1. **Jangan latih kalau tidak perlu** - pakai model kecil/fine-tuning/distilasi
   atau API yang sudah ada; sebagian besar kebutuhan kampus/industri kecil
   selesai dengan model < 1B parameter.
2. **Kurangi beban inferensi** - kuantisasi (int8/4-bit), pruning, caching,
   batching, dan routing ke model kecil untuk query mudah.
3. **Ukur dan laporkan** - catat energi/emisi per eksperimen (mis. wandb +
   CodeCarbon), jadikan metrik seperti akurasi. Yang tidak diukur tidak
   dikurangi; ini juga alasan wandb relevan di praktikum.
4. **Pemilihan waktu & lokasi** - penjadwalan job ke jam/region dengan
   intensitas karbon listrik rendah, serta PPA energi terbarukan.
5. **Pendinginan & air** - pendingin cair tertutup/immersion dan daur ulang
   air, bukan sekali pakai.
6. **Umur pakai & daur ulang** - perpanjang masa pakai akselerator, tuntut
   desain yang bisa diperbaiki, dan kelola e-waste resmi.
7. **Regulasi & transparansi** - kewajiban pelaporan konsumsi energi model
   besar, seperti arah EU AI Act terhadap model tujuan umum.

Referensi:
- IEA (2025), *Energy and AI*, https://www.iea.org/reports/energy-and-ai
- Strubell, Ganesh & McCallum (2019), *Energy and Policy Considerations for
  Deep Learning in NLP*, ACL 2019, https://aclanthology.org/P19-1355/
- Bender, Gebru, McMillan-Major & Shmitchell (2021), *On the Dangers of
  Stochastic Parrots*, FAccT 2021, https://doi.org/10.1145/3442188.3445922
- Schwartz dkk. (2020), *Green AI*, Communications of the ACM,
  https://doi.org/10.1145/3386890
- Microsoft, *Responsible AI - Principles and Approach* (prinsip
  reliability/safety & accountability),
  https://www.microsoft.com/en-us/ai/principles-and-approach
