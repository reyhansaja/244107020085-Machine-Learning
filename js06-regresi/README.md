# JS06 - Model Regresi

Pengerjaan jobsheet **JS06 Regresi** (JTI Polinema 2025/2026):
https://polinema.gitbook.io/jti-modul-praktikum-pembelajaran-mesin-2025-2026/js06-regresi/praktikum1

## Struktur

```
js06-regresi/
├── data/
│   ├── dataset.csv        # Ecommerce Customers (500 baris) - praktikum 1
│   ├── Posisi_gaji.csv    # Posisi vs Gaji (10 baris) - praktikum 2
│   └── insurance.csv      # Medical Cost Personal (1338 baris) - tugas
├── praktikum1/regresi_linier_sederhana.ipynb   # OLS statsmodels + analisis residual
├── praktikum2/svr_posisi_gaji.ipynb            # SVR RBF (regresi non-linier)
├── tugas/multiple_linear_regression.ipynb      # MLR + pembanding SVR & polinomial
└── <praktikum>/output/*.png|*.csv
```

Dataset diunduh langsung dari lampiran GitBook (bukan `/files/<id>`, tapi URL
`3041032130-files.gitbook.io/...&token=...` hasil grep HTML halaman).

## Cara jalan

```bash
cd c:/code/pemmes26
.venv/Scripts/python.exe js06-regresi/praktikum1/regresi_linier_sederhana.ipynb
.venv/Scripts/python.exe js06-regresi/praktikum2/svr_posisi_gaji.ipynb
.venv/Scripts/python.exe js06-regresi/tugas/multiple_linear_regression.ipynb
```

## Hasil (output nyata)

**Praktikum 1** - regresi sederhana `Yearly Amount Spent ~ Length of Membership`
(train:test = 70:30, random_state=100):

| Metrik | Nilai |
|---|---|
| intercept | 265.2483 |
| slope | 66.3015 |
| R² train | 0.6688 |
| R² test | 0.6119 |
| residual std | 46.69 |

Sama persis dengan modul (265.2483 + 66.3015x, R² test 0.612) karena dataset dan
random_state identik.

**Praktikum 2** - SVR RBF pada `Posisi_gaji.csv`:

| Metrik | Nilai |
|---|---|
| Prediksi gaji level 6.5 | 170370.02 |
| MAE / MSE / RMSE / R² (skala terstandardisasi) | 0.222993 / 0.248400 / 0.498397 / 0.751600 |
| MAE / RMSE / R² (skala asli) | 63332.39 / 141550.32 / 0.751600 |

Identik dengan modul (170370.0204065; MAE 0.222993; R² 0.7516). Metrik skala
asli ditambahkan supaya errornya terbaca dalam satuan gaji.

**Tugas** - Multiple Linear Regression `charges ~ age + bmi + children + sex + smoker + region`
(test 20%, random_state=42, one-hot drop first):

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Multiple Linear Regression | 4181.19 | 5796.28 | 0.7836 |
| Polinomial derajat 2 | 4254.27 | 5841.28 | 0.7802 |
| SVR RBF (C=1000, eps=1000) | 3895.33 | 7782.40 | 0.6099 |

Koefisien terbesar: `smoker_yes` +23651 (perokok naik ~$23.6k/tahun), lalu
`children` +425 dan `bmi` +337 per satuan.

## Deviasi dari modul & sebabnya

1. `sns.distplot` (deprecated, hilang di seaborn 0.13) diganti `sns.histplot(kde=True)`;
   `pairplot(size=...)` diganti `height=...` (parameter baru). Lingkungan pakai seaborn 0.13.2.
2. Garis regresi praktikum 1 digambar dari `lr.params` hasil fit, bukan angka
   hardcode `265.2483 + 66.3015*X` seperti di modul (angkanya tetap sama, hanya
   tidak lagi hardcode).
3. Praktikum 2: modul memplot X dan y yang sudah di-scale; di sini kurva SVR
   di-inverse-transform dulu sehingga sumbu = level posisi & gaji asli. Metrik
   versi scaled tetap dicetak agar bisa dibandingkan langsung dengan modul.
4. Tugas: modul hanya minta MLR; ditambah pembanding SVR dan polinomial derajat 2
   sebagai analisis. SVR kalah (R² 0.61 vs 0.78) karena `charges` sangat
   heteroskedastik dan punya outlier ~$63k, sementara SVR RBF dengan parameter
   default kurang toleran; percobaan log-transform target juga tidak menaikkan R².
