"""
Enhancement Nomor Ijazah
Metode: Brightness Adjustment, Contrast Stretching, Histogram Equalization
Fitur:
- Memilih/crop area nomor ijazah dengan mouse
- Menampilkan citra sebelum dan sesudah enhancement
- Menampilkan histogram sebelum dan sesudah
- Menyimpan hasil gambar dan grafik
- Mencoba OCR jika Tesseract sudah terpasang
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

# Jika Tesseract terpasang tetapi tidak terdeteksi, buka komentar baris di bawah
# dan sesuaikan dengan lokasi tesseract.exe di komputer kamu.
# Contoh:
# import pytesseract
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def pilih_gambar():
    """Membuka jendela untuk memilih gambar ijazah."""
    root = Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Pilih gambar ijazah",
        filetypes=[
            ("File gambar", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
            ("Semua file", "*.*")
        ]
    )
    root.destroy()
    return path


def brightness_adjustment(gray, nilai=30):
    """Menaikkan kecerahan sebesar nilai (0-255)."""
    return cv2.convertScaleAbs(gray, alpha=1.0, beta=nilai)


def contrast_stretching(gray):
    """Meregangkan intensitas piksel ke rentang 0-255."""
    minimum = int(np.min(gray))
    maksimum = int(np.max(gray))

    if maksimum == minimum:
        return gray.copy()

    hasil = (gray.astype(np.float32) - minimum) * (255.0 / (maksimum - minimum))
    return np.clip(hasil, 0, 255).astype(np.uint8)


def histogram_equalization(gray):
    """Menerapkan histogram equalization pada citra grayscale."""
    return cv2.equalizeHist(gray)


def tampilkan_hasil(nama, asli, hasil, folder_output):
    """Menampilkan citra asli/hasil beserta histogram dan menyimpan grafik."""
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    ax[0, 0].imshow(asli, cmap="gray", vmin=0, vmax=255)
    ax[0, 0].set_title("Sebelum Enhancement")
    ax[0, 0].axis("off")

    ax[0, 1].imshow(hasil, cmap="gray", vmin=0, vmax=255)
    ax[0, 1].set_title(f"Sesudah: {nama}")
    ax[0, 1].axis("off")

    ax[1, 0].hist(asli.ravel(), bins=256, range=(0, 256), color="gray")
    ax[1, 0].set_title("Histogram Sebelum")
    ax[1, 0].set_xlabel("Intensitas piksel")
    ax[1, 0].set_ylabel("Jumlah piksel")

    ax[1, 1].hist(hasil.ravel(), bins=256, range=(0, 256), color="gray")
    ax[1, 1].set_title(f"Histogram Sesudah: {nama}")
    ax[1, 1].set_xlabel("Intensitas piksel")
    ax[1, 1].set_ylabel("Jumlah piksel")

    fig.tight_layout()
    nama_file = nama.lower().replace(" ", "_")
    fig.savefig(os.path.join(folder_output, f"perbandingan_{nama_file}.png"), dpi=150)
    plt.show()
    plt.close(fig)


def jalankan_ocr(nama, citra):
    """Menjalankan OCR jika pytesseract dan mesin Tesseract tersedia."""
    try:
        import pytesseract
        konfigurasi = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789./-"
        teks = pytesseract.image_to_string(citra, config=konfigurasi).strip()
        print(f"Hasil OCR ({nama}): {teks if teks else '[teks tidak terdeteksi]'}")
        return teks
    except Exception as error:
        print(f"OCR ({nama}) dilewati: {error}")
        return "OCR tidak tersedia"


def main():
    path = pilih_gambar()
    if not path:
        print("Tidak ada gambar yang dipilih. Program dihentikan.")
        return

    gambar = cv2.imread(path)
    if gambar is None:
        print("Gambar tidak dapat dibaca. Periksa format dan lokasi file.")
        return

    gray = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)

    print("\nPilih area nomor ijazah dengan mouse.")
    print("Tarik kotak mengelilingi nomor ijazah, lalu tekan ENTER atau SPACE.")
    print("Tekan C untuk membatalkan crop dan memakai seluruh gambar.")
    x, y, w, h = cv2.selectROI("Crop area nomor ijazah", gray, showCrosshair=True)
    cv2.destroyAllWindows()

    if w > 0 and h > 0:
        area = gray[int(y):int(y + h), int(x):int(x + w)]
    else:
        area = gray

    nama_folder = os.path.splitext(os.path.basename(path))[0] + "_hasil_enhancement"
    folder_output = os.path.join(os.path.dirname(path), nama_folder)
    os.makedirs(folder_output, exist_ok=True)

    hasil_metode = {
        "Brightness Adjustment": brightness_adjustment(area, nilai=30),
        "Contrast Stretching": contrast_stretching(area),
        "Histogram Equalization": histogram_equalization(area),
    }

    cv2.imwrite(os.path.join(folder_output, "area_asli.png"), area)

    print("\n=== HASIL OCR ===")
    print("Catatan: hasil OCR bergantung pada kualitas gambar dan instalasi Tesseract.")
    jalankan_ocr("Original", area)

    for nama, hasil in hasil_metode.items():
        nama_file = nama.lower().replace(" ", "_")
        cv2.imwrite(os.path.join(folder_output, f"{nama_file}.png"), hasil)
        tampilkan_hasil(nama, area, hasil, folder_output)
        jalankan_ocr(nama, hasil)

    print(f"\nSelesai. Semua hasil disimpan di:\n{folder_output}")


if __name__ == "__main__":
    main()
