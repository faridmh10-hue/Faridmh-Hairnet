# app.py
import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

# --- 1. Definisi Konstanta dan Path ---
IMAGE_SIZE = (150, 150)
CLASSES = ['curly', 'dreadlocks', 'kinky', 'straight', 'wavy']

# Mapping label kelas dari indeks numerik ke nama string
# Pastikan urutan ini sesuai dengan train_generator.class_indices saat pelatihan
# Misalnya, jika train_generator.class_indices menghasilkan {'curly':0, 'dreadlocks':1, ...}
CLASS_LABELS = {i: cls for i, cls in enumerate(CLASSES)}

# --- 2. Rekomendasi Perawatan Rambut ---
HAIR_CARE_RECOMMENDATIONS = {
    'curly': "Rambut keriting membutuhkan hidrasi ekstra. Gunakan produk bebas sulfat, kondisioner yang kaya, dan rutin menggunakan masker rambut. Hindari menyisir rambut saat kering dan gunakan teknik 'plop' untuk mengeringkan.",
    'dreadlocks': "Dreadlocks membutuhkan pembersihan rutin untuk menghindari penumpukan produk dan menjaga kebersatan. Gunakan sampo khusus dreadlocks dan pastikan untuk mengeringkan sepenuhnya untuk mencegah jamur. Minyak alami seperti tea tree oil dapat membantu menjaga kesehatan kulit kepala.",
    'kinky': "Rambut kinky sangat kering dan rapuh. Fokus pada pelembapan intensif dengan leave-in conditioner, minyak alami (kelapa, shea butter), dan teknik 'LOC' (Liquid, Oil, Cream). Lindungi rambut di malam malam dengan satin bonnet atau sarung bantal sutra.",
    'straight': "Rambut lurus cenderung berminyak lebih cepat. Gunakan sampo yang membersihkan tanpa membuat kering dan kondisioner ringan. Hindari produk berlebihan yang bisa membuat rambut lepek. Keramas secara teratur dan sisir rambut dengan lembut.",
    'wavy': "Rambut bergelombang berada di antara lurus dan keriting. Gunakan produk yang ringan untuk menambah volume dan definisi tanpa membuat rambut berat. Scrunch rambut saat basah untuk membentuk gelombang dan biarkan kering secara alami atau gunakan diffuser."
}

# --- 3. Memuat Model ---
MODEL_PATH = 'hair_type_classifier_model.h5' # Sesuaikan nama file model Anda
# PASTIKAN NAMA FILE INI SAMA PERSIS DENGAN NAMA FILE .h5 MODEL ANDA

LOADED_MODEL = None # Inisialisasi di luar try/except

try:
    # Memuat model tanpa mengkompilasinya terlebih dahulu
    LOADED_MODEL = load_model(MODEL_PATH, compile=False)
    # Setelah dimuat, Anda bisa mengkompilasinya kembali jika diperlukan untuk inference,
    # atau jika model memerlukan optimizer untuk berfungsi (meski umumnya tidak untuk inference).
    # Untuk kasus klasifikasi sederhana, seringkali tidak perlu kompilasi ulang.
    # LOADED_MODEL.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("Model berhasil dimuat.")
except Exception as e:
    print(f"Gagal memuat model: {e}. Pastikan file model '{MODEL_PATH}' ada di direktori yang sama.")
    # LOADED_MODEL = None # Sudah diinisialisasi di awal, jadi tidak perlu lagi di sini

# --- 4. Fungsi Prediksi ---
def predict_hair_type(image_array):
    if LOADED_MODEL is None:
        return "Model gagal dimuat. Harap periksa file model."

    try:
        # Konversi numpy array gambar ke PIL Image, lalu resize dan konversi ke RGB
        img = Image.fromarray(image_array).convert('RGB')
        img = img.resize(IMAGE_SIZE)
        img_processed = np.array(img) / 255.0
        img_processed = np.expand_dims(img_processed, axis=0) # Tambahkan dimensi batch

        predictions = LOADED_MODEL.predict(img_processed)[0]

        # Mengurutkan prediksi dari persentase tertinggi
        sorted_predictions_indices = np.argsort(predictions)[::-1]

        results = {}
        for i in sorted_predictions_indices:
            label = CLASS_LABELS[i]
            percentage = predictions[i] * 100
            results[label] = percentage

        # Tipe rambut dengan persentase tertinggi
        predicted_class_index = np.argmax(predictions)
        predicted_label = CLASS_LABELS[predicted_class_index]

        recommendation = HAIR_CARE_RECOMMENDATIONS.get(predicted_label, "Tidak ada rekomendasi spesifik untuk tipe rambut ini.")

        output_string = "### Hasil Analisis Tipe Rambut:\n\n"
        for label, percentage in results.items():
            output_string += f"- **{label.capitalize()}**: {percentage:.2f}%\n"

        output_string += f"\n---"
        output_string += f"\n### Tipe Rambut Teridentifikasi: **{predicted_label.capitalize()}**"
        output_string += f"\n\n### Rekomendasi Perawatan:\n"
        output_string += recommendation

        return output_string

    except Exception as e:
        return f"Terjadi kesalahan saat prediksi: {e}"

# --- 5. Antarmuka Gradio ---
if LOADED_MODEL is not None:
    # Ganti 'interface' menjadi 'app'
    app = gr.Interface(
        fn=predict_hair_type,
        inputs=gr.Image(type="numpy", label="Unggah Gambar Model Rambut"),
        outputs=gr.Markdown(label="Hasil Prediksi dan Rekomendasi"),
        title="Klasifikasi Tipe Rambut dan Rekomendasi Perawatan",
        description="Unggah gambar model rambut untuk mendapatkan prediksi tipe rambut (Curly, Dreadlocks, Kinky, Straight, Wavy) dan rekomendasi perawatan.",
        examples=[ # Opsional: Tambahkan contoh gambar untuk dicoba
            # Contoh gambar ini perlu ada di folder 'examples' di Spaces Anda
            # atau ganti dengan URL gambar publik jika ingin langsung dari internet
            # "./examples/curly_example.jpg",
            # "./examples/straight_example.jpg"
        ]
    )
    # Tambahkan baris ini untuk mengaktifkan antrean Gradio
    app.queue()

    # Baris ini harus tetap dikomentari
    # if __name__ == "__main__":
    #    interface.launch()
else:
    print("Tidak dapat membuat antarmuka Gradio karena model gagal dimuat.")