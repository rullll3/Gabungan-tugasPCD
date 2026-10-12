import io
import os
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import cv2

st.set_page_config(page_title="Image Enhancement Citra Ijazah", layout="wide")

st.title("Gambar Enhanment Citra Ijazah")
st.write(
    "Unggah satu atau beberapa citra ijazah. Program mengubah RGB menjadi grayscale, "
    "menampilkan histogram, menganalisis intensitas, lalu membandingkan beberapa metode enhancement."
)

st.info(
    "Analisis otomatis di bawah ini adalah perkiraan berdasarkan statistik piksel. "
    "Periksa hasil secara visual, terutama keterbacaan teks dan detail ijazah."
)

uploaded_files = st.file_uploader(
    "Pilih gambar ijazah (PNG, JPG, JPEG, BMP, TIFF)",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
    accept_multiple_files=True
)

method_options = [
    "Otomatis (berdasarkan kondisi citra)",
    "CLAHE",
    "Gamma Correction",
    "Histogram Equalization",
    "Contrast Stretching",
    "Brightness Adjustment"
]
selected_method = st.selectbox("Metode enhancement", method_options)
gamma_value = st.slider("Nilai gamma (untuk Gamma Correction)", 0.2, 2.5, 0.8, 0.1)
brightness_value = st.slider("Penyesuaian kecerahan (untuk Brightness Adjustment)", -100, 100, 20, 5)
clip_limit = st.slider("Clip limit CLAHE", 1.0, 5.0, 2.0, 0.5)
tile_size = st.slider("Ukuran grid CLAHE", 4, 16, 8, 1)

def pil_to_rgb(pil_img):
    return np.array(ImageOps.exif_transpose(pil_img).convert("RGB"))

def rgb_to_gray(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

def get_analysis(gray):
    mean = float(np.mean(gray))
    std = float(np.std(gray))
    p5, p95 = np.percentile(gray, [5, 95])
    dynamic_range = float(p95 - p5)

    # Perkiraan masalah berdasarkan statistik global; bukan diagnosis mutlak.
    issues = []
    if mean < 85:
        issues.append("Cenderung terlalu gelap (rata-rata intensitas rendah).")
    elif mean > 190:
        issues.append("Cenderung terlalu terang (rata-rata intensitas tinggi).")
    else:
        issues.append("Kecerahan rata-rata berada pada rentang menengah.")

    if dynamic_range < 90 or std < 35:
        issues.append("Kemungkinan kontras rendah (sebaran intensitas relatif sempit).")
    else:
        issues.append("Rentang intensitas cukup lebar; tetap periksa detail secara visual.")

    # Perkiraan pencahayaan tidak merata dari perbedaan rata-rata pada beberapa area.
    h, w = gray.shape
    patch_means = []
    for y0, y1 in [(0, h // 2), (h // 2, h)]:
        for x0, x1 in [(0, w // 2), (w // 2, w)]:
            patch = gray[y0:y1, x0:x1]
            if patch.size:
                patch_means.append(float(np.mean(patch)))
    if patch_means and (max(patch_means) - min(patch_means) > 45):
        issues.append("Kemungkinan pencahayaan tidak merata (perbedaan kecerahan antararea besar).")
    else:
        issues.append("Statistik area tidak menunjukkan ketimpangan pencahayaan yang sangat besar.")

    # Indikator kasar noise: variasi residual terhadap median blur.
    median = cv2.medianBlur(gray, 3)
    residual = cv2.absdiff(gray, median)
    noise_score = float(np.mean(residual))
    if noise_score > 12:
        issues.append(f"Ada indikasi detail/noise berfrekuensi tinggi (skor residual {noise_score:.1f}).")
    else:
        issues.append(f"Indikator residual tinggi relatif rendah (skor {noise_score:.1f}); noise tetap perlu diperiksa visual.")

    return {
        "mean": mean,
        "std": std,
        "p5": float(p5),
        "p95": float(p95),
        "dynamic_range": dynamic_range,
        "noise_score": noise_score,
        "issues": issues,
    }

def apply_enhancement(gray, method):
    if method == "CLAHE":
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        return clahe.apply(gray), "CLAHE membantu meningkatkan kontras lokal dan dapat membantu saat pencahayaan tidak merata."
    if method == "Gamma Correction":
        normalized = gray.astype(np.float32) / 255.0
        corrected = np.power(normalized, gamma_value) * 255.0
        return np.clip(corrected, 0, 255).astype(np.uint8), (
            "Gamma Correction dengan gamma < 1 mencerahkan area gelap; gamma > 1 menggelapkan citra."
        )
    if method == "Histogram Equalization":
        return cv2.equalizeHist(gray), "Histogram Equalization menyebarkan intensitas global, tetapi dapat memperkuat noise."
    if method == "Contrast Stretching":
        lo, hi = np.percentile(gray, [2, 98])
        if hi <= lo:
            return gray.copy(), "Rentang intensitas terlalu sempit untuk diregangkan secara bermakna."
        out = (gray.astype(np.float32) - lo) * (255.0 / (hi - lo))
        return np.clip(out, 0, 255).astype(np.uint8), "Contrast Stretching meregangkan rentang intensitas berdasarkan persentil 2–98."
    if method == "Brightness Adjustment":
        out = np.clip(gray.astype(np.int16) + brightness_value, 0, 255).astype(np.uint8)
        return out, "Brightness Adjustment menambah atau mengurangi nilai intensitas secara langsung."

    stats = get_analysis(gray)
    mean = stats["mean"]
    if mean < 85:
        out = np.clip(gray.astype(np.float32) / 255.0, 0, 1)
        out = np.power(out, 0.65) * 255
        return out.astype(np.uint8), "Otomatis: Gamma Correction untuk membantu mencerahkan citra gelap."
    if mean > 190:
        out = np.clip(gray.astype(np.float32) / 255.0, 0, 1)
        out = np.power(out, 1.25) * 255
        return out.astype(np.uint8), "Otomatis: Gamma Correction untuk membantu mengurangi kecerahan berlebih."
    if stats["dynamic_range"] < 90 or stats["std"] < 35:
        return cv2.equalizeHist(gray), "Otomatis: Histogram Equalization karena rentang intensitas/kontras relatif rendah."
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size)).apply(gray), (
        "Otomatis: CLAHE untuk meningkatkan kontras lokal tanpa meratakan seluruh citra secara berlebihan."
    )

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def make_histogram(gray, title):
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.hist(gray.ravel(), bins=256, range=(0, 256), color="gray")
    ax.set_title(title)
    ax.set_xlabel("Intensitas piksel (0–255)")
    ax.set_ylabel("Jumlah piksel")
    ax.set_xlim(0, 255)
    ax.grid(alpha=0.2)
    return fig

if uploaded_files:
    os.makedirs("hasil_enhancement", exist_ok=True)
    for file in uploaded_files:
        try:
            pil_img = Image.open(file)
            rgb = pil_to_rgb(pil_img)
            gray = rgb_to_gray(rgb)
            enhanced, reason = apply_enhancement(gray, selected_method)
            stats = get_analysis(gray)

            st.header(f"File: {file.name}")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("Citra RGB asli")
                st.image(rgb, use_container_width=True)
            with col2:
                st.subheader("Grayscale")
                st.image(gray, clamp=True, use_container_width=True)
            with col3:
                st.subheader("Hasil enhancement")
                st.image(enhanced, clamp=True, use_container_width=True)

            h1, h2 = st.columns(2)
            with h1:
                st.pyplot(make_histogram(rgb_to_gray(rgb), "Histogram citra asli (grayscale dari RGB)"))
            with h2:
                st.pyplot(make_histogram(enhanced, "Histogram setelah enhancement"))

            st.subheader("Analisis distribusi intensitas")
            a, b, c, d = st.columns(4)
            a.metric("Rata-rata intensitas", f"{stats['mean']:.2f}")
            b.metric("Standar deviasi", f"{stats['std']:.2f}")
            c.metric("Rentang P5–P95", f"{stats['dynamic_range']:.2f}")
            d.metric("Skor residual", f"{stats['noise_score']:.2f}")

            st.markdown("**Identifikasi kemungkinan masalah:**")
            for issue in stats["issues"]:
                st.write(f"- {issue}")

            st.markdown("**Metode dan alasan:**")
            st.write(reason)

            st.download_button(
                label=f"Unduh grayscale — {file.name}",
                data=cv2.imencode(".png", gray)[1].tobytes(),
                file_name=f"{os.path.splitext(file.name)[0]}_grayscale.png",
                mime="image/png"
            )
            st.download_button(
                label=f"Unduh hasil enhancement — {file.name}",
                data=cv2.imencode(".png", enhanced)[1].tobytes(),
                file_name=f"{os.path.splitext(file.name)[0]}_enhanced.png",
                mime="image/png"
            )
            st.divider()
        except Exception as e:
            st.error(f"Gagal memproses {file.name}: {e}")
else:
    st.warning("Unggah minimal satu gambar untuk memulai.")
    st.markdown(
        "Tips: gunakan citra ijazah yang berbeda kondisinya, misalnya gelap, terang, kontras rendah, "
        "atau memiliki bayangan. Jangan mengunggah dokumen pribadi ke layanan publik jika tidak diperlukan."
    )

st.caption("Catatan: histogram RGB asli dihitung dari versi grayscale agar distribusi intensitas dapat dibandingkan dengan histogram grayscale.")
