
import os
import re
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytesseract
from tkinter import Tk, filedialog, messagebox

def pilih_gambar():
    """Memilih citra nomor ijazah dari komputer."""
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Pilih gambar nomor ijazah",
        filetypes=[
            ("File gambar", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
            ("Semua file", "*.*"),
        ],
    )
    root.destroy()
    return path


def crop_area(gray):
    """Memilih area nomor ijazah. Jika crop dibatalkan, gunakan seluruh gambar."""
    print("\nPilih area nomor ijazah dengan mouse.")
    print("Tarik kotak di sekitar nomor, lalu tekan ENTER/SPACE.")
    print("Tekan C untuk memakai seluruh gambar.")
    try:
        x, y, w, h = cv2.selectROI(
            "Crop area nomor ijazah", gray, showCrosshair=True, fromCenter=False
        )
        cv2.destroyAllWindows()
        if w > 0 and h > 0:
            return gray[int(y):int(y + h), int(x):int(x + w)]
    except cv2.error:
        print("Jendela crop OpenCV tidak tersedia. Gambar penuh akan digunakan.")
    finally:
        cv2.destroyAllWindows()
    return gray


def mean_filter(gray):
    return cv2.blur(gray, (3, 3))


def median_filter(gray):
    # Ukuran kernel median harus ganjil dan lebih besar dari 1.
    return cv2.medianBlur(gray, 3)


def gaussian_filter(gray):
    return cv2.GaussianBlur(gray, (3, 3), 0)


def sharpening(gray):
    """Unsharp masking sederhana untuk memperjelas tepi karakter."""
    blur = cv2.GaussianBlur(gray, (0, 0), 2)
    return cv2.addWeighted(gray, 1.7, blur, -0.7, 0)


def jalankan_ocr(citra):
    """OCR dengan konfigurasi yang sesuai untuk satu baris nomor."""
    konfigurasi = (
        "--oem 3 --psm 7 "
        "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789./-"
    )
    try:
        return pytesseract.image_to_string(citra, config=konfigurasi).strip()
    except pytesseract.TesseractNotFoundError:
        return "[Tesseract belum terinstal atau belum terhubung]"
    except Exception as error:
        return f"[OCR gagal: {error}]"


def normalisasi_teks(teks):
    """Menyisakan karakter alfanumerik untuk perbandingan."""
    return re.sub(r"[^A-Za-z0-9]", "", teks).upper()


def hitung_karakter_benar(teks_ocr, teks_referensi):
    """
    Menghitung karakter benar berdasarkan posisi (perbandingan karakter per karakter).
    Untuk pengukuran yang adil, masukkan nomor referensi yang benar saat program berjalan.
    """
    hasil = normalisasi_teks(teks_ocr)
    referensi = normalisasi_teks(teks_referensi)
    if not referensi:
        return 0, 0, 0.0
    benar = sum(a == b for a, b in zip(hasil, referensi))
    akurasi = benar / len(referensi) * 100
    return benar, len(referensi), akurasi


def simpan_histogram(citra, judul, path_output):
    plt.figure(figsize=(8, 4))
    plt.hist(citra.ravel(), bins=256, range=(0, 256), color="gray")
    plt.title(f"Histogram - {judul}")
    plt.xlabel("Intensitas piksel")
    plt.ylabel("Jumlah piksel")
    plt.tight_layout()
    plt.savefig(path_output, dpi=150)
    plt.close()


def simpan_perbandingan(citra_asli, hasil_metode, folder_output):
    """Simpan gambar asli dan hasil tiap metode dalam satu panel perbandingan."""
    nama_metode = list(hasil_metode.keys())
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()

    axes[0].imshow(citra_asli, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original")
    axes[0].axis("off")

    for index, nama in enumerate(nama_metode, start=1):
        axes[index].imshow(hasil_metode[nama], cmap="gray", vmin=0, vmax=255)
        axes[index].set_title(nama)
        axes[index].axis("off")

    axes[5].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(folder_output, "perbandingan_semua_metode.png"), dpi=150)
    plt.show()
    plt.close(fig)


def main():
    path = pilih_gambar()
    if not path:
        print("Tidak ada gambar yang dipilih. Program dihentikan.")
        return

    gambar = cv2.imread(path)
    if gambar is None:
        print("Gambar tidak dapat dibaca. Periksa file yang dipilih.")
        return

    gray = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)
    area_asli = crop_area(gray)
    if area_asli.size == 0:
        print("Area gambar kosong. Program dihentikan.")
        return

    nama_folder = os.path.splitext(os.path.basename(path))[0] + "_hasil_filtering_ocr"
    folder_output = os.path.join(os.path.dirname(path), nama_folder)
    os.makedirs(folder_output, exist_ok=True)

    # Metode enhancement yang diminta dalam tugas.
    hasil_metode = {
        "Mean Filter": mean_filter(area_asli),
        "Median Filter": median_filter(area_asli),
        "Gaussian Filter": gaussian_filter(area_asli),
        "Sharpening": sharpening(area_asli),
    }

    cv2.imwrite(os.path.join(folder_output, "original.png"), area_asli)
    simpan_histogram(area_asli, "Original", os.path.join(folder_output, "histogram_original.png"))

    print("\n=== PERBANDINGAN OCR ===")
    print("Masukkan nomor ijazah yang benar untuk menghitung karakter benar dan akurasi.")
    print("Contoh format: 571012022000056")
    referensi = input("Nomor referensi (kosongkan jika tidak ingin menghitung akurasi): ").strip()

    baris = []
    teks_original = jalankan_ocr(area_asli)
    if referensi:
        benar, total, akurasi = hitung_karakter_benar(teks_original, referensi)
        karakter = f"{benar}/{total}"
        akurasi_tampil = f"{akurasi:.2f}%"
    else:
        karakter, akurasi_tampil = "-", "-"

    baris.append({
        "Metode": "Original",
        "Hasil OCR": teks_original or "[teks tidak terdeteksi]",
        "Karakter Benar": karakter,
        "Akurasi": akurasi_tampil,
    })
    print(f"Original: {teks_original or '[teks tidak terdeteksi]'}")

    for nama, hasil in hasil_metode.items():
        nama_file = nama.lower().replace(" ", "_")
        cv2.imwrite(os.path.join(folder_output, f"{nama_file}.png"), hasil)
        simpan_histogram(
            hasil, nama, os.path.join(folder_output, f"histogram_{nama_file}.png")
        )
        teks = jalankan_ocr(hasil)
        print(f"{nama}: {teks or '[teks tidak terdeteksi]'}")

        if referensi:
            benar, total, akurasi = hitung_karakter_benar(teks, referensi)
            karakter = f"{benar}/{total}"
            akurasi_tampil = f"{akurasi:.2f}%"
        else:
            karakter, akurasi_tampil = "-", "-"

        baris.append({
            "Metode": nama,
            "Hasil OCR": teks or "[teks tidak terdeteksi]",
            "Karakter Benar": karakter,
            "Akurasi": akurasi_tampil,
        })

    simpan_perbandingan(area_asli, hasil_metode, folder_output)

    tabel = pd.DataFrame(baris)
    csv_path = os.path.join(folder_output, "tabel_perbandingan_ocr.csv")
    xlsx_path = os.path.join(folder_output, "tabel_perbandingan_ocr.xlsx")
    tabel.to_csv(csv_path, index=False, encoding="utf-8-sig")
    try:
        tabel.to_excel(xlsx_path, index=False)
    except Exception as error:
        print(f"File Excel tidak dibuat: {error}")

    print("\n=== TABEL OCR ===")
    print(tabel.to_string(index=False))
    print(f"\nSelesai. Hasil tersimpan di:\n{folder_output}")


if __name__ == "__main__":
    main()
