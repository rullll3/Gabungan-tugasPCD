# Mini Project Image Enhancement Citra Ijazah

## 1. Persiapan
Pastikan Python sudah terpasang. Buka folder proyek ini di VS Code.

## 2. Buat virtual environment (disarankan)
Buka Terminal VS Code pada folder proyek, lalu jalankan:

```bash
python -m venv venv
```

Aktifkan virtual environment di Windows:

```bash
venv\Scripts\activate
```

Jika PowerShell memblokir aktivasi, gunakan Command Prompt terminal atau lanjutkan dengan Python/pip yang tersedia.

## 3. Instal library
Pastikan terminal berada di folder yang memiliki `requirements.txt`, kemudian jalankan:

```bash
python -m pip install -r requirements.txt
```

## 4. Jalankan aplikasi
Jalankan perintah:

```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser. Jika tidak terbuka otomatis, salin alamat lokal yang muncul di terminal (biasanya `http://localhost:8501`).

## 5. Cara menggunakan
1. Klik area unggah gambar dan pilih satu atau beberapa citra ijazah.
2. Pilih metode enhancement. Pilih **Otomatis** jika ingin program memilih metode berdasarkan statistik awal.
3. Periksa citra RGB, grayscale, hasil enhancement, dan histogram.
4. Baca bagian analisis intensitas dan alasan metode.
5. Unduh hasil grayscale dan enhancement untuk dokumentasi laporan.

## Metode yang tersedia
- CLAHE: meningkatkan kontras lokal.
- Gamma Correction: menyesuaikan kecerahan nonlinier.
- Histogram Equalization: meningkatkan kontras global.
- Contrast Stretching: meregangkan rentang intensitas.
- Brightness Adjustment: menambah/mengurangi kecerahan.

## Catatan analisis
Deteksi gelap/terang, kontras, pencahayaan tidak merata, dan noise adalah indikator awal berbasis statistik, bukan keputusan mutlak. Pastikan memeriksa hasil secara visual. Histogram yang lebih tersebar tidak otomatis berarti citra lebih baik; detail teks, noise, dan keterbacaan tetap harus diperhatikan.
