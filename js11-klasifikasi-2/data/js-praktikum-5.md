> For the complete documentation index, see [llms.txt](https://polinema.gitbook.io/jti-modul-praktikum-pembelajaran-mesin-2025-2026/llms.txt). Markdown versions of documentation pages are available by appending `.md` to page URLs; this page is available as [Markdown](https://polinema.gitbook.io/jti-modul-praktikum-pembelajaran-mesin-2025-2026/js11-klasifikasi-2/praktikum-5.md).

# Praktikum 5

Klasifikasi Citra Siang dan Malang

## Pengantar

Pada percobaan ini kita akan mencoba melakukan klasifikasi pada dua label citra, yaitu siang dan malam. Percobaan ini akan memberikan pengalaman bagi Anda untuk melakukan proses pra pengolahan data, ekstraksi fitur, dan melakukan klasifikasi dengan menggunakan *classifier* sederhana dan SVM.

Unduh dataset berikut,

{% file src="/files/W1t7Ny1MQ5z3Echvhz2N" %}

## Langkah 0 - Import Library

{% code overflow="wrap" lineNumbers="true" %}

```python
# Import Required Libraries
from pathlib import Path
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import random
import numpy as np
import pandas as pd
```

{% endcode %}

Lakukan ekstraksi data gambar, kemudian definisikan lokasi gambar. Pada contoh ini, folder gambar berlokasi sama dengan lokasi file python.

{% code overflow="wrap" lineNumbers="true" %}

```python
# Image directories
train_dir = "images/training/"
test_dir = "images/test/"
```

{% endcode %}

## Langkah 1 - Load Data dan Visualisasikan

Buatlah fungsi untuk membuat daftar seluruh path dari gambar.

{% code overflow="wrap" lineNumbers="true" %}

```python
def load_dataset(img_dir):
    p = Path(img_dir)
    dirs = p.glob('*')

    img_list = []

    for dir in dirs:
        label = str(dir).split('/')[-1]
        for file in dir.glob('*.jpg'):
            img = mpimg.imread(file)

            if not img is None:
                img_list.append((img, label))
    
    return img_list
```

{% endcode %}

Load gambar training

{% code overflow="wrap" lineNumbers="true" %}

```python
# Load training data
train_img = load_dataset(train_dir)
```

{% endcode %}

Lakukan pengecekan pada salah satu data pada list. List harus berisi tuple dengan dua data, yaitu data gambar dan label dari gambar.

{% code overflow="wrap" lineNumbers="true" %}

```python
# Check the first data
# It should be a tuple consist of arrays of image and image labels
train_img[0]
```

{% endcode %}

Cek ukuran gambar secara acak

{% code overflow="wrap" lineNumbers="true" %}

```python
# Random size checking
pick_random = np.random.randint(0, len(train_img))

# Check img size
print(f'Image {pick_random}')
print(train_img[pick_random][0].shape)
```

{% endcode %}

Tampilkan gambar untuk inspeksi secara visual. Buatlah fungsi untuk membantu memvisualkan gambar

{% code overflow="wrap" lineNumbers="true" %}

```python
# Function to Visualize
def random_img_viz(img_list):
    rand_num = np.random.randint(0, len(img_list))

    img = img_list[rand_num][0]
    label = img_list[rand_num][1]
    label_str = 'day' if label == 1 else 'night'

    plt.imshow(img)
    print(f'Shape\t: {img.shape}')
    print(f'Label\t: {label}')
```

{% endcode %}

Lakukan visualisasi gambar secara acak

{% code overflow="wrap" lineNumbers="true" %}

```python
random_img_viz(train_img)
```

{% endcode %}

Jika fungsi visualisasi berjalan dengan benar, maka akan muncul tampilan seperti berikut,

<figure><img src="https://3041032130-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F5CvtE8Xh9b75jKUaRr5Y%2Fuploads%2FG2yb8Nitc7EOAxPMB9XV%2Fimage.png?alt=media&amp;token=43bcfb91-3a21-4b90-9f79-5acbfb892fa7" alt=""><figcaption></figcaption></figure>

## Langkah 3 - Pra Pengolahan Data

Pada tahap ini, kita akan melakukan dua proses utama, yaitu standardisasi ukuran gambar, dan *encoding* label gambar.

Bualah fungsi berikut untuk menstandarkan ukuran gambar.

{% code overflow="wrap" lineNumbers="true" %}

```python
def standarized_input(image):
    # resize to w: 1100, h:600
    std_img = cv2.resize(image, (1100,600))

    return std_img
```

{% endcode %}

Selanjutnya, buatlah fungsi untuk kebutuhan *encoding* label.

{% code overflow="wrap" lineNumbers="true" %}

```python
def label_encoder(label):
    # Encode the label
    # day as 1; night as 0
    num_val = 0

    if(label == 'day'):
        num_val = 1
    
    return num_val
```

{% endcode %}

Terakhir, buatlah fungsi untuk melakukan kedua hal tersebut secara sekaligus untuk semua gambar dalam list.

{% code overflow="wrap" lineNumbers="true" %}

```python
def preprocess(img_list):
    std_img_list = []

    for item in img_list:
        image = item[0]
        label = item[1]

        # Standarized the image
        std_img = standarized_input(image)

        # Create the label
        img_label = label_encoder(label)

        std_img_list.append((std_img, img_label))
    
    return std_img_list
```

{% endcode %}

Lakukan pra pengolahan data pada data training.

{% code overflow="wrap" lineNumbers="true" %}

```python
train_std_img_list = preprocess(train_img)
```

{% endcode %}

Lakukan pengecekan ukuran gambar secara acak.

{% code overflow="wrap" lineNumbers="true" %}

```python
# Random size checking
pick_random = np.random.randint(0, len(train_std_img_list))

# Check img size
print(f'Image {pick_random}')
print(train_std_img_list[pick_random][0].shape)
```

{% endcode %}

Anda akan mendapatkan output seperti berikut,

```
Image 189
(600, 1100, 3)
```

**WARNING!** Perhatikan ukuran (shape) dari data image. Atribut shape akan menampilkan dalam konteks baris (height) x kolom (width)

Lakukan inspeksi visual dengan fungsi `random_img_viz` yang telah dibuat sebelumnya pada gambar hasil pra pengolahan.

Hasilnya akan seperti gambar berikut,

<figure><img src="https://3041032130-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F5CvtE8Xh9b75jKUaRr5Y%2Fuploads%2FZsUxl8rDhNFBumJnQchu%2Fimage.png?alt=media&amp;token=f5aa6246-a33d-46cd-b67b-14c04102d527" alt=""><figcaption></figcaption></figure>

## Langkah 4 - Ekstraksi Fitur

Selanjutnya, untuk dapat membadakan antara label satu dengan label yang lain, kita memerlukan sebuah fitur. Fitur adalah penciri dari sebuah data yang dapat digunakan untuk membendakan data satu dengan yang lain. Pada percobaan kali ini, kita akan menggunakan fitur yang sederhana yaitu nilai rata-rata dari tingkat kecerahan gambar (average brightness). Namun sebelum dapat melakukan hal tersebut, kita akan mengubah ruang warna (colorspace) dari RGB menjadi HSV (Hue, Saturation, Value). Hal ini dikarenakan, tingkat kecerahan (brightness) lebih mudah didapatkan dari HSV berdasarkan nilai Valuenya.

Buatlah fungsi berikut untuk mendapatkan nilai rata-rata tingkat kecerahan.

{% code overflow="wrap" lineNumbers="true" %}

```python
# Get feature based on average brightness using HSV colorspace
def avg_brightness(image):
    # Convert image to HSV
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Calculate the avg of brightness
    sum_brightness = np.sum(img_hsv[:,:,2]) # take the 3rb value which is the V channel
    area = image.shape[0] * image.shape[1]
    avg = sum_brightness / area

    return avg
```

{% endcode %}

Lakukan pengecekan pada gambar secara acak. **INGAT! Gunakan gambar yang telah melalui proses pra pengolahan data!**

{% code overflow="wrap" lineNumbers="true" %}

```python
# Check on random image
rand_img = np.random.randint(0, len(train_std_img_list))

feature_img = train_std_img_list[rand_img][0]

avg_img = avg_brightness(feature_img)

print(f'Image {rand_img}')
print(f'Avg Brighness: {avg_img:.4f}')
plt.imshow(feature_img)
```

{% endcode %}

Akan didapatkan hasil output seperti pada gambar.

<figure><img src="https://3041032130-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F5CvtE8Xh9b75jKUaRr5Y%2Fuploads%2F5swfp4f5XfUAxll35anW%2Fimage.png?alt=media&amp;token=17e1dd45-4e32-4c5a-9a15-e9c99744c036" alt=""><figcaption></figcaption></figure>

## Langkah 5 - Klasifikasi dengan Metode Threshold

Pada tahap ini, kita akan melakukan proses klasifikasi sederhana dengan menggunakan nilai ambang batas (threshold) dari nilai rata-rata kecerahan yang kita tentukan sendiri.

Buatlah fungsi berikut,

{% code overflow="wrap" lineNumbers="true" %}

```python
def predict_label(img, threshold):
    # Computer average brightness
    avg = avg_brightness(img)
    pred = 0

    # Predict the label based on user defined threshold
    if avg > threshold:
        pred = 1
    
    return pred
```

{% endcode %}

Lakukan pengecekan prediksi secara acak pada data training

{% code overflow="wrap" lineNumbers="true" %}

```python
# Test the classifier on train data
rand_img = np.random.randint(0, len(train_std_img_list))

pred = predict_label(train_std_img_list[rand_img][0], threshold=120)

# Evaluate
print(f'Image {rand_img}')
print(f'Actual label: {train_std_img_list[rand_img][1]}')
print(f'Predicted label: {pred}')
plt.imshow(train_std_img_list[rand_img][0])
```

{% endcode %}

Hasilnya akan seperti pada gambar

<figure><img src="https://3041032130-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F5CvtE8Xh9b75jKUaRr5Y%2Fuploads%2F3fdnQeVumjIJGSua9Gh6%2Fimage.png?alt=media&amp;token=422b32ff-dbed-428b-b9a1-7de6c1f37740" alt=""><figcaption></figcaption></figure>

## Langkah 6 - Evaluasi Manual

Selanjutnya kita akan membuat fungsi evaluasi model sederhana, yaitu dengan membandingkan label yang diprediksi benar dengan seluruh data. Ingat kembali konsep ***confussion matrix***.

Buatlah fungsi berikut,

{% code overflow="wrap" lineNumbers="true" %}

```python
def evaluate(img_list, threshold):
    miss_labels = []

    for file in img_list:
        # Get the ground truth / correct label
        img = file[0]
        label = file[1]

        # Get prediction
        pred_label = predict_label(img, threshold)

        # Compare ground truth and pred
        if pred_label != label:
            miss_labels.append((img, pred_label, label))
    
    total_img = len(img_list)
    corr_pred = total_img - len(miss_labels)
    accuracy = corr_pred / total_img

    print(f'Accuracy: {accuracy:.4f}')
```

{% endcode %}

Lakukan evaluasi pada data training dengan nilai ambang batas 120.

{% code overflow="wrap" lineNumbers="true" %}

```python
# Evaluate on train data
evaluate(train_std_img_list, threshold=120)
```

{% endcode %}

Hasilnya adalah,

```
Accuracy: 0.8417
```

Anda dapat mengubah nilai ambang batas dan amati hasilnya.

Selanjutnya, kita akan melakukan evaluasi pada data testing. Namun sebelumnya, data testing harus diperlakukan sama dengan data training dalam konteks pra progolahan data dan ekstraksi fitur.

{% code overflow="wrap" lineNumbers="true" %}

```python
# Evaluate on test data

# Load test data
test_img = load_dataset(test_dir)

# Preprocess
test_std_img_list = preprocess(test_img)

# Predict
evaluate(test_std_img_list, threshold=120)
```

{% endcode %}

Hasil akurasi dari data testing adalah,

```
Accuracy: 0.8688
```

***

## Klasifikasi dengan SVM

Sebelumnya, kita hanya menggunakan threshold sebagai acuan. Cara ini mungkin tidak efektif dikarenakan kita harus menentukan threshold dengan tepat. Oleh karena itu, selanjutnya kita akan mencoba menggunakan model SVM untuk proses klasifikasi. Seluruh langkah yang digunakan serupa, kita hanya mengubah mulai langkah ke-4.

## Langkah 4 Alternatif - Membuat Feature Vectors.

Perbedaan mendasar dari langkah 4 sebelumnya adalah, kita akan melakukan tabulasi semua nilai rata-rata kecerahan pada data, dan menyimpannya dalam bentuk tabel. Dalam konteks ini, kita akan membuat tabel dengan kolom fitur dan label.

Buatlah fungsi berikut,

{% code overflow="wrap" lineNumbers="true" %}

```python
# Create function to extract feature for every images and stored in tabular data
# Stored in Pandas dataframe
def extract_avg_bright_feature(img_list):
    avg_list = []
    labels = []

    for img in img_list:
        img_avg = avg_brightness(img[0]) # Get the avg brightness from image
        img_label = img[1] # Get the image label

        avg_list.append(img_avg)
        labels.append(img_label)
    
    # Stack data in columcular way
    data = np.column_stack((avg_list, labels))
    # Create a Pandas dataframe
    df = pd.DataFrame(data, columns=['AVG_BRIGHT', 'LABELS'])

    return df
```

{% endcode %}

Cek hasilnya pada data training,

{% code overflow="wrap" lineNumbers="true" %}

```python
# Extract feature on train data
train_avg_img = extract_avg_bright_feature(train_std_img_list)
print(f'Shape: {train_avg_img.shape}')
train_avg_img.head()
```

{% endcode %}

Maka akan tampil output seperti pada gambar

<figure><img src="https://3041032130-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F5CvtE8Xh9b75jKUaRr5Y%2Fuploads%2FRdmkZJVrqBzNfUK5FoT4%2Fimage.png?alt=media&amp;token=807f0618-25c8-469e-95d9-7e7609b82909" alt=""><figcaption></figcaption></figure>

Lakukan langkah yang serupa pada data testing

{% code overflow="wrap" lineNumbers="true" %}

```python
# Do the same thing on test data
test_avg_img = extract_avg_bright_feature(test_std_img_list)
print(f'Shape: {test_avg_img.shape}')
test_avg_img.head()
```

{% endcode %}

Hasilnya adalah seperti pada gambar

<figure><img src="https://3041032130-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F5CvtE8Xh9b75jKUaRr5Y%2Fuploads%2FzMIP08hHGWFICIDbMlz1%2Fimage.png?alt=media&amp;token=a7df1254-846d-4926-921f-ad5038a89e9f" alt=""><figcaption></figcaption></figure>

## Langkah 5 - Buat Model SVM

Selanjutnya, kita akan membuat model SVM dengan kernel RBF (default) dengan memanfaatkan libary scikit-learn.

{% code overflow="wrap" lineNumbers="true" %}

```python
# import requied library
from sklearn.svm import SVC

# Split data and label
X_train = train_avg_img.iloc[:,0].values.reshape(-1,1)
y_train = train_avg_img.iloc[:,1]
X_test = test_avg_img.iloc[:,0].values.reshape(-1,1)
y_test = test_avg_img.iloc[:,1]

model = SVC()
model.fit(X_train, y_train)
```

{% endcode %}

## Langkah 6 - Evaluasi

Selanjutnya, kita akan melakukan evaluasi pada data training dan testing dengan bantuan library scikit-learn.

{% code overflow="wrap" lineNumbers="true" %}

```python
from sklearn.metrics import accuracy_score

# Make a prediction on train data
y_train_pred = model.predict(X_train)

# Get the accuracy on train data
acc_train = accuracy_score(y_train, y_train_pred)

# Make a prediction on test data
y_test_pred = model.predict(X_test)

# Get the accuracy on test data
acc_test = accuracy_score(y_test, y_test_pred)

# Print Eval Result
print(f'Accuracy on train: {acc_train}')
print(f'Accuracy on test: {acc_test}')
```

{% endcode %}

Hasil akurasi dengan model SVM adalah,

```
Accuracy on train: 0.8583333333333333
Accuracy on test: 0.9
```

## Acknowledgment

[Arunn Thevapalan](https://github.com/arunnthevapalan) in [Day Night Image Classifier on Github](https://github.com/arunnthevapalan/day-night-classifier/tree/master).
