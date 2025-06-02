# 🧠 Sistem Prediksi Tingkat Stres Mahasiswa dengan Logika Fuzzy

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](#)
[![Status](https://img.shields.io/badge/status-Finished-brightgreen)](#)

> Prediksi tingkat stres mahasiswa berdasarkan **beban tugas** dan **jam tidur** menggunakan **Fuzzy Inference System** metode **Mamdani** dengan **defuzzifikasi Centroid**.

---

## 📂 Deskripsi Proyek

Sistem ini dirancang untuk memprediksi tingkat stres mahasiswa menggunakan pendekatan **logika fuzzy**. Dengan dua input utama—**beban tugas** dan **jam tidur**—sistem dapat mengklasifikasikan stres menjadi tiga kategori: **Rendah**, **Sedang**, dan **Tinggi**.

Pendekatan ini menggunakan:
- **Metode Mamdani** sebagai mesin inferensi fuzzy
- **Metode Centroid** untuk defuzzifikasi
- Implementasi dalam **Python** menggunakan pustaka `scikit-fuzzy`

---

## 📌 Fitur Utama

✅ Fuzzifikasi input linguistik (beban tugas & jam tidur)  
✅ 9 aturan IF–THEN berbasis logika fuzzy  
✅ Defuzzifikasi menggunakan metode **Centroid**  
✅ Prediksi tingkat stres dalam bentuk numerik & kategori  
✅ Visualisasi fungsi keanggotaan dan output sistem

---

## 🛠️ Teknologi & Library

- Python `3.12`
- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)
- [scikit-fuzzy](https://pythonhosted.org/scikit-fuzzy/)

---

## 🚀 Cara Menjalankan

1. **Clone repository**
```bash
gh repo clone Antares023/fuzzy_tingkat_stress_mahasiswa.git
cd fuzzy_tingkat_stress_mahasiswa
```

2. **Install Library yg Dibutuhkan**
```bash
pip install "ketik library yang dibutuhkan (tanpa tanda petik)"
```

3. **Jalankan Program**
```bash
python fuzzy_tingkat_stress_mahasiswa.py
```

---

## 🧪 Contoh Output

```bash
Input: Beban Tugas = 6, Jam Tidur = 5
Output Numerik (Prediksi): 72.3
Kategori: Tinggi
```

---

## 📄 Lisensi

Proyek ini dilisensikan di bawah MIT License.

---

## 👨‍💻 Kontributor

🧑 Muhammad Ilham Ramdhani
🎓 Universitas Muhammadiyah Cirebon – Teknik Informatika

---

## 📬 Kontak

📧 Email: ilhamrmdhnii02@gmail.com
