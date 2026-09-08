import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ----------------------------------------------------
# Step 1: Data Preprocessing (Part 1)
# ----------------------------------------------------
print("--- STEP 1: DATA PREPROCESSING ---")
# 1. Load Dataset
dataset_path = 'circuits.csv'
df = pd.read_csv(dataset_path, encoding='latin1')

print(f"Dataset shape: {df.shape}")
print(df[['circuitId', 'name', 'country', 'lat', 'lng', 'alt']].head())

# 2. Handling Missing Values
print("\nMissing values count:")
print(df.isnull().sum())

# 'alt' has mostly missing values (72 missing), so we drop it or impute if needed.
df_clean = df.drop(columns=['alt', 'url', 'circuitRef'])

# 3. Feature Selection
# Independent variable (X): Latitude ('lat')
# Dependent variable (y): Longitude ('lng')
X = df_clean[['lat']].values
y = df_clean['lng'].values

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------------------------------
# Step 2: Polynomial Regression Algorithm
# ----------------------------------------------------
print("\n--- STEP 2: POLYNOMIAL REGRESSION ---")
degree = 3  # Degree of Polynomial

# Polynomial Feature Transformation
poly_reg = PolynomialFeatures(degree=degree)
X_poly_train = poly_reg.fit_transform(X_train)
X_poly_test = poly_reg.transform(X_test)

# Feature Scaling
scaler = StandardScaler()
X_poly_train_scaled = scaler.fit_transform(X_poly_train)
X_poly_test_scaled = scaler.transform(X_poly_test)

# Model Training
model = LinearRegression()
model.fit(X_poly_train_scaled, y_train)

# Model Prediction
y_pred = model.predict(X_poly_test_scaled)

# ----------------------------------------------------
# Step 3: Evaluation & Outcome
# ----------------------------------------------------
print("\n--- STEP 3: MODEL EVALUATION & SIMPLE OUTCOME ---")
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Polynomial Degree: {degree}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {np.sqrt(mse):.2f}")
print(f"R-squared Score (R2): {r2:.4f}")

# Plotting the regression curve
X_grid = np.arange(X.min(), X.max(), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
X_grid_poly = poly_reg.transform(X_grid)
X_grid_poly_scaled = scaler.transform(X_grid_poly)

plt.figure(figsize=(9, 5))
plt.scatter(X, y, color='red', label='Actual Data (Lat vs Lng)')
plt.plot(X_grid, model.predict(X_grid_poly_scaled), color='blue', linewidth=2, label=f'Polynomial Fit (Degree {degree})')
plt.title('Polynomial Regression Model Outcome')
plt.xlabel('Latitude (lat)')
plt.ylabel('Longitude (lng)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('polynomial_regression_outcome.png')
print("\nPlot saved successfully as 'polynomial_regression_outcome.png'.")
