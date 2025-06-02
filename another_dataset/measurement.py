import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import seaborn as sns

# 1. Load dan Visualisasi Data
print("1. ANALISIS DATA AWAL")
data = pd.read_csv('Student Mental health.csv')

# Bersihkan data tahun studi
data['Your current year of Study'] = data['Your current year of Study'].str.lower().str.replace(' ', '')


# 2. Sistem Fuzzy
print("\n2. MEMBANGUN SISTEM FUZZY")

# Variabel input/output
cgpa = ctrl.Antecedent(np.arange(0, 4.5, 0.1), 'cgpa')  # Range CGPA 0-4
tahun = ctrl.Antecedent(np.arange(1, 5, 1), 'tahun')    # Tahun studi 1-4
stres = ctrl.Consequent(np.arange(0, 101, 1), 'stres')  # Tingkat stres 0-100

# Fungsi keanggotaan CGPA
cgpa['Rendah'] = fuzz.trimf(cgpa.universe, [0, 0, 2.5])
cgpa['Sedang'] = fuzz.trimf(cgpa.universe, [2.0, 3.0, 4.0])
cgpa['Tinggi'] = fuzz.trimf(cgpa.universe, [3.5, 4.0, 4.0])

# Fungsi keanggotaan tahun studi
tahun['Awal'] = fuzz.trimf(tahun.universe, [1, 1, 2])
tahun['Tengah'] = fuzz.trimf(tahun.universe, [1, 2.5, 4])
tahun['Akhir'] = fuzz.trimf(tahun.universe, [3, 4, 4])

# Fungsi keanggotaan stres (Gaussian)
stres['Rendah'] = fuzz.gaussmf(stres.universe, 20, 10)
stres['Sedang'] = fuzz.gaussmf(stres.universe, 50, 10)
stres['Tinggi'] = fuzz.gaussmf(stres.universe, 80, 10)

# Visualisasi fungsi keanggotaan
cgpa.view()
plt.suptitle('Fungsi Keanggotaan CGPA')
plt.show()

tahun.view()
plt.suptitle('Fungsi Keanggotaan Tahun Studi')
plt.show()

stres.view()
plt.suptitle('Fungsi Keanggotaan Tingkat Stres')
plt.show()

# 3. Aturan Fuzzy
print("\n3. ATURAN FUZZY")
rule1 = ctrl.Rule(cgpa['Rendah'] & tahun['Awal'], stres['Rendah'])
rule2 = ctrl.Rule(cgpa['Rendah'] & tahun['Tengah'], stres['Rendah'])
rule3 = ctrl.Rule(cgpa['Rendah'] & tahun['Akhir'], stres['Sedang'])
rule4 = ctrl.Rule(cgpa['Sedang'] & tahun['Awal'], stres['Sedang'])
rule5 = ctrl.Rule(cgpa['Sedang'] & tahun['Tengah'], stres['Sedang'])
rule6 = ctrl.Rule(cgpa['Sedang'] & tahun['Akhir'], stres['Tinggi'])
rule7 = ctrl.Rule(cgpa['Tinggi'] & tahun['Awal'], stres['Tinggi'])
rule8 = ctrl.Rule(cgpa['Tinggi'] & tahun['Tengah'], stres['Tinggi'])
rule9 = ctrl.Rule(cgpa['Tinggi'] & tahun['Akhir'], stres['Tinggi'])

# 4. Sistem Kontrol
stres_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
stres_sim = ctrl.ControlSystemSimulation(stres_ctrl)

# 5. Prediksi untuk 5 Mahasiswa
print("\n4. CONTOH PREDIKSI UNTUK 5 MAHASISWA")

def cgpa_to_value(cgpa_str):
    if '0 - 1.99' in cgpa_str: return 1.0
    elif '2.00 - 2.49' in cgpa_str: return 2.25
    elif '2.50 - 2.99' in cgpa_str: return 2.75
    elif '3.00 - 3.49' in cgpa_str: return 3.25
    elif '3.50 - 4.00' in cgpa_str: return 3.75
    else: return 2.5

def tahun_to_value(tahun_str):
    if 'year1' in tahun_str: return 1
    elif 'year2' in tahun_str: return 2
    elif 'year3' in tahun_str: return 3
    elif 'year4' in tahun_str: return 4
    else: return 1

for i in range(10):
    mhs = data.iloc[i]
    cgpa_val = cgpa_to_value(mhs['What is your CGPA?'])
    tahun_val = tahun_to_value(mhs['Your current year of Study'])
    
    stres_sim.input['cgpa'] = cgpa_val
    stres_sim.input['tahun'] = tahun_val
    stres_sim.compute()
    
    print(f"\nMahasiswa {i+1}:")
    print(f"Gender: {mhs['Choose your gender']}, Umur: {mhs['Age']}")
    print(f"Tahun: {mhs['Your current year of Study']}, CGPA: {mhs['What is your CGPA?']}")
    print(f"Depresi: {mhs['Do you have Depression?']}, Anxiety: {mhs['Do you have Anxiety?']}, Panic Attack: {mhs['Do you have Panic attack?']} Treatment: {mhs['Did you seek any specialist for a treatment?']}")
    print(f"Prediksi Stres: {stres_sim.output['stres']:.2f}")

    plt.title(f'Prediksi Stres Mahasiswa {i+1}')