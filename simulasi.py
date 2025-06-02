import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Variabel input dan output
beban_tugas = ctrl.Antecedent(np.arange(0, 11, 1), 'beban_tugas')
jam_tidur = ctrl.Antecedent(np.arange(0, 9, 1), 'jam_tidur')
tingkat_stres = ctrl.Consequent(np.arange(0, 101, 1), 'tingkat_stres')

# Fungsi keanggotaan untuk beban tugas
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

# Aturan fuzzy
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

# Contoh penggunaan
stres_simulator.input['beban_tugas'] = 2  # Beban tugas sedang-berat
stres_simulator.input['jam_tidur'] = 8    # Jam tidur cukup

# Hitung hasil
stres_simulator.compute()

# Hasil defuzzifikasi dengan COA
print("Tingkat stres:", stres_simulator.output['tingkat_stres'])
tingkat_stres.view(sim=stres_simulator)
plt.title('Hasil Prediksi Tingkat Stres')
plt.show()