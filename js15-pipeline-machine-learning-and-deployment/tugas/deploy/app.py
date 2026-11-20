import numpy as np
import cv2
import pickle
import tensorflow as tf
from flask import Flask, request, render_template_string

app = Flask(__name__)

MODEL_PATH = "digit_model.h5"
SCALER_PATH = "digit_scaler.pkl"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    print("✅ System Loaded Successfully")
except Exception as e:
    print(f"❌ Error loading system: {e}")


def preprocess_image(image_bytes):
    # Decode gambar -> grayscale -> 8x8 (sama dengan ukuran training) -> 0..1
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("File bukan gambar yang valid")
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
    feat = (img.astype("float32") / 255.0).reshape(1, -1)
    return scaler.transform(feat)


@app.route("/", methods=["GET"])
def home():
    return render_template_string("""
    <div style="text-align:center; padding:50px;">
        <h1>Digit Classifier (0-9)</h1>
        <p>Unggah citra digit tulisan tangan (latar gelap, digit terang).</p>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="file" required><br><br>
            <button type="submit">Prediksi</button>
        </form>
    </div>
    """)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["file"]
        data = preprocess_image(file.read())
        prob = model.predict(data, verbose=0)[0]
        digit = int(np.argmax(prob))
        return (f"<h2 style='text-align:center'>Hasil: {digit} "
                f"(confidence {prob[digit]*100:.2f}%)</h2>"
                f"<center><a href='/'>Kembali</a></center>")
    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    # Port 7860 wajib untuk Hugging Face Spaces
    app.run(host="0.0.0.0", port=7860)
