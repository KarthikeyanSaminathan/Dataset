"""
Simple Machine Learning Model for circuits.csv
Task: Predict Geographic Region (Americas, Europe, Asia-Oceania, Africa) 
      from GPS Coordinates (Latitude, Longitude).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("=========================================================")
print("          SIMPLE ML PREDICTION MODEL (CIRCUITS)          ")
print("=========================================================")

# 1. Load Data
df = pd.read_csv('circuits.csv', encoding='latin1')
print(f"Loaded {len(df)} circuit records from circuits.csv.\n")

# 2. Map Country to Geographic Region (Target Variable)
europe = {'UK', 'France', 'Italy', 'Germany', 'Spain', 'Belgium', 'Monaco', 
          'Hungary', 'Austria', 'Netherlands', 'Portugal', 'Sweden', 
          'Switzerland', 'Russia', 'Azerbaijan', 'Turkey'}
americas = {'USA', 'Canada', 'Brazil', 'Argentina', 'Mexico'}
asia_oceania = {'Australia', 'Malaysia', 'Bahrain', 'China', 'Singapore', 
                'Japan', 'Korea', 'India', 'UAE'}
africa = {'South Africa', 'Morocco'}

def get_region(country):
    if country in europe:
        return 'Europe'
    elif country in americas:
        return 'Americas'
    elif country in asia_oceania:
        return 'Asia-Oceania'
    elif country in africa:
        return 'Africa'
    return 'Other'

df['region'] = df['country'].apply(get_region)

# 3. Define Features (X) and Target (y)
X = df[['lat', 'lng']]
y = df['region']

# 4. Train / Test Split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate Outcome
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"--- MODEL OUTCOME ---")
print(f"Model: Random Forest Classifier")
print(f"Prediction Accuracy: {accuracy * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 7. Sample Prediction Test
print("--- TEST PREDICTIONS ON NEW COORDINATES ---")
samples = pd.DataFrame([
    {'name': 'Silverstone, UK', 'lat': 52.0786, 'lng': -1.01694},
    {'name': 'Circuit of the Americas, USA', 'lat': 30.1328, 'lng': -97.6411},
    {'name': 'Suzuka, Japan', 'lat': 34.8431, 'lng': 136.541},
    {'name': 'Kyalami, South Africa', 'lat': -25.9989, 'lng': 28.0767}
])

sample_preds = model.predict(samples[['lat', 'lng']])
for i, row in samples.iterrows():
    print(f"Track: {row['name']} (lat: {row['lat']}, lng: {row['lng']}) -> Predicted Region: {sample_preds[i]}")

# 8. Save Outcome Visualization
plt.figure(figsize=(10, 5))
colors = {'Europe': 'blue', 'Americas': 'red', 'Asia-Oceania': 'green', 'Africa': 'orange'}

for region, color in colors.items():
    subset = df[df['region'] == region]
    plt.scatter(subset['lng'], subset['lat'], label=region, color=color, s=50, alpha=0.8, edgecolors='black')

plt.title('Circuits Machine Learning Outcome: GPS Coordinates to Region', fontsize=12)
plt.xlabel('Longitude (lng)')
plt.ylabel('Latitude (lat)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('simple_model_outcome.png', dpi=150)

print("\nOutcome chart saved to 'simple_model_outcome.png'.")
print("=========================================================")
