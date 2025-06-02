import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================
# 1. Baca Dataset
# ==============================================
df = pd.read_csv('mental_health.csv')  # Pastikan file ada di direktori yang sama
print("Preview Dataset:")
print(df.head(100))

# ==============================================
# 2. Sistem Fuzzy
# ==============================================
# Variabel Input/Output
beban_tugas = ctrl.Antecedent(np.arange(1, 11, 1), 'beban_tugas')
jam_tidur = ctrl.Antecedent(np.arange(1, 11, 1), 'jam_tidur')
tingkat_stres = ctrl.Consequent(np.arange(0, 101, 1), 'tingkat_stres')

# Fungsi Keanggotaan
beban_tugas['Ringan'] = fuzz.trimf(beban_tugas.universe, [0, 0, 5])
beban_tugas['Sedang'] = fuzz.trimf(beban_tugas.universe, [3, 5, 7])
beban_tugas['Berat'] = fuzz.trimf(beban_tugas.universe, [5, 10, 10])

jam_tidur['Sedikit'] = fuzz.trimf(jam_tidur.universe, [0, 0, 5])
jam_tidur['Cukup'] = fuzz.trimf(jam_tidur.universe, [3, 5, 7])
jam_tidur['Banyak'] = fuzz.trimf(jam_tidur.universe, [5, 10, 10])

tingkat_stres['Rendah'] = fuzz.trimf(tingkat_stres.universe, [0, 0, 50])
tingkat_stres['Sedang'] = fuzz.trimf(tingkat_stres.universe, [30, 50, 70])
tingkat_stres['Tinggi'] = fuzz.trimf(tingkat_stres.universe, [50, 100, 100])

# Visualisasi fungsi keanggotaan
beban_tugas.view()
plt.title('Fungsi Keanggotaan Beban Tugas')
plt.show()

jam_tidur.view()
plt.title('Fungsi Keanggotaan Jam Tidur')
plt.show()

tingkat_stres.view()
plt.title('Fungsi Keanggotaan Tingkat Stres')
plt.show()

# Aturan Fuzzy
rule1 = ctrl.Rule(beban_tugas['Ringan'] & jam_tidur['Banyak'], tingkat_stres['Rendah'])
rule2 = ctrl.Rule(beban_tugas['Ringan'] & jam_tidur['Cukup'], tingkat_stres['Rendah'])
rule3 = ctrl.Rule(beban_tugas['Ringan'] & jam_tidur['Sedikit'], tingkat_stres['Sedang'])
rule4 = ctrl.Rule(beban_tugas['Sedang'] & jam_tidur['Banyak'], tingkat_stres['Rendah'])
rule5 = ctrl.Rule(beban_tugas['Sedang'] & jam_tidur['Cukup'], tingkat_stres['Sedang'])
rule6 = ctrl.Rule(beban_tugas['Sedang'] & jam_tidur['Sedikit'], tingkat_stres['Tinggi'])
rule7 = ctrl.Rule(beban_tugas['Berat'] & jam_tidur['Banyak'], tingkat_stres['Sedang'])
rule8 = ctrl.Rule(beban_tugas['Berat'] & jam_tidur['Cukup'], tingkat_stres['Tinggi'])
rule9 = ctrl.Rule(beban_tugas['Berat'] & jam_tidur['Sedikit'], tingkat_stres['Tinggi'])

# Sistem kontrol
stres_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
stres_simulator = ctrl.ControlSystemSimulation(stres_ctrl)

# ==============================================
# 3. Prediksi untuk Seluruh Dataset
# ==============================================
def predict_stres(beban, tidur):
    stres_simulator.input['beban_tugas'] = beban
    stres_simulator.input['jam_tidur'] = tidur
    stres_simulator.compute()
    return stres_simulator.output['tingkat_stres']

df['Prediksi_Stres'] = df.apply(lambda row: predict_stres(row['Beban_Tugas'], row['Jam_Tidur']), axis=1)
df['Kategori_Prediksi'] = pd.cut(df['Prediksi_Stres'], bins=[0, 40, 70, 100], labels=['Rendah', 'Sedang', 'Tinggi'])

# ==============================================
# 4. Visualisasi dengan Seaborn
# ==============================================
plt.figure(figsize=(12, 6))

# Scatter Plot: Beban Tugas vs Jam Tidur (Warna berdasarkan Prediksi)
sns.scatterplot(
    data=df, 
    x='Beban_Tugas', 
    y='Jam_Tidur', 
    hue='Kategori_Prediksi',
    palette={'Rendah': 'green', 'Sedang': 'orange', 'Tinggi': 'red'},
    s=100
)
plt.title('Hubungan Beban Tugas, Jam Tidur, dan Tingkat Stres (Prediksi Fuzzy)')
plt.xlabel('Beban Tugas (1-10)')
plt.ylabel('Jam Tidur (Jam)')
plt.grid(True)
plt.legend(title='Tingkat Stres')

plt.tight_layout()
plt.show()

# ==============================================
# 5. Tampilkan Hasil Prediksi
# ==============================================
print("\nHasil Prediksi untuk 5 Data Pertama:")
print(df[['Nama', 'Beban_Tugas', 'Jam_Tidur', 'Tingkat_Stres', 'Prediksi_Stres', 'Kategori_Prediksi']].head(100))