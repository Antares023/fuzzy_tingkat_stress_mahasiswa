import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
data = pd.read_csv('Student Mental health.csv')

# Preprocessing data
data['Depression'] = data['Do you have Depression?'].map({'Yes': 1, 'No': 0})
data['Anxiety'] = data['Do you have Anxiety?'].map({'Yes': 1, 'No': 0})
data['Panic_attack'] = data['Do you have Panic attack?'].map({'Yes': 1, 'No': 0})

# Create total symptoms score
data['Symptoms'] = data['Depression'] + data['Anxiety'] + data['Panic_attack']

# Map CGPA to numerical score (0-100 scale)
cgpa_mapping = {
    '0 - 1.99': 30,
    '2.00 - 2.49': 50,
    '2.50 - 2.99': 65,
    '3.00 - 3.49': 80,
    '3.50 - 4.00': 95
}
data['Academic_Score'] = data['What is your CGPA?'].map(cgpa_mapping)

# Create mental health risk categories
def get_risk_level(row):
    if row['Symptoms'] == 0:
        return 'Low Risk'
    elif row['Symptoms'] == 1:
        return 'Moderate Risk' 
    else:
        return 'High Risk'

data['Risk_Level'] = data.apply(get_risk_level, axis=1)

# Create the plot
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")

# Scatter plot with color by risk level
scatter = sns.scatterplot(
    data=data,
    x='Academic_Score',
    y='Symptoms',
    hue='Risk_Level',
    palette={'Low Risk':'#2ecc71', 'Moderate Risk':'#f39c12', 'High Risk':'#e74c3c'},
    s=100,
    alpha=0.8
)

# Add titles and labels
plt.title('Student Mental Health Risk vs Academic Performance', pad=20, fontsize=16)
plt.xlabel('Academic Performance Score (CGPA converted to 0-100 scale)', fontsize=12)
plt.ylabel('Number of Mental Health Symptoms', fontsize=12)

# Adjust axes
plt.xlim(0, 100)
plt.ylim(-0.5, 3.5)
plt.xticks(np.arange(0, 101, 10))
plt.yticks([0, 1, 2, 3])

# Add reference lines
plt.axhline(y=1.5, color='gray', linestyle='--', alpha=0.3)
plt.axvline(x=80, color='gray', linestyle='--', alpha=0.3)

# Add annotations
plt.text(82, 3.2, 'High Performers', alpha=0.7)
plt.text(82, -0.3, 'Low Risk Zone', alpha=0.7)
plt.text(30, 3.2, 'High Risk Students', alpha=0.7)

# Improve legend
plt.legend(title='Mental Health Risk Level', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show plot
plt.tight_layout()
plt.show()