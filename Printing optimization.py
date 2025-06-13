import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
import statsmodels.api as sm
import xgboost as xgb

# ===========================
# Load Dataset
# ===========================
data_path = '/Users/amiralibozorgnia/Library/CloudStorage/GoogleDrive-amiralibozorgnia79@gmail.com/My Drive/my computer/ML/printing.csv'
printer = pd.read_csv(data_path)

# Encode categorical variables
printer['infill_pattern'].replace(['grid', 'honeycomb'], [0, 1], inplace=True)
printer['material'].replace(['abs', 'pla'], [0, 1], inplace=True)

# ===========================
# Feature Engineering
# ===========================
X_base = printer.drop(['roughness', 'tension_strenght', 'elongation'], axis=1)
y = printer['roughness']

# Add interaction terms
X = X_base.copy()
X['inter_mn'] = X['material'] * X['nozzle_temperature']
X['inter_bn'] = X['bed_temperature'] * X['nozzle_temperature']
X['inter_fn'] = X['fan_speed'] * X['nozzle_temperature']
X['inter_fb'] = X['fan_speed'] * X['bed_temperature']

# ===========================
# Linear Regression via StatsModels
# ===========================
def linear_regression_report(X, y):
    X_ols = sm.add_constant(X)
    model = sm.OLS(y, X_ols).fit()
    print(model.summary())

print("Linear Regression Summary (StatsModels):")
linear_regression_report(X, y)

# ===========================
# Scale and Split Data
# ===========================
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ===========================
# Train Models
# ===========================
models = {
    "Linear Regression (sklearn)": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=6, min_samples_leaf=2, random_state=42),
    "Decision Tree": DecisionTreeRegressor(max_depth=3, min_samples_split=4, random_state=42),
    "XGBoost": xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4,
                                subsample=0.9, colsample_bytree=0.9, random_state=42)
}

r2_results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    r2_results[name] = r2
    print(f"{name}\n  R²:   {r2:.3f}\n")

# ===========================
# Results Summary Table
# ===========================
r2_df = pd.DataFrame.from_dict(r2_results, orient='index', columns=['R2 Score'])

# ===========================
# Visualization
# ===========================
plt.figure(figsize=(7, 5))
sns.barplot(x=r2_df.index, y=r2_df['R2 Score'])
plt.title("R2 Score Comparison Across Models")
plt.xticks(rotation=30)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# ===========================
# Optional: Correlation Heatmap
# ===========================
plt.figure(figsize=(12, 6))
sns.heatmap(printer.corr(), annot=True, cmap='coolwarm', linewidths=1)
plt.title("Correlation Heatmap of 3D Printing Dataset")
plt.show()

# ===========================
# 📌 Save R² Score Comparison Plot
# ===========================
plt.figure(figsize=(7, 5))
sns.barplot(x=r2_df.index, y=r2_df['R2 Score'])
plt.title("R2 Score Comparison Across Models")
plt.xticks(rotation=30)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("r2_score_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

# ===========================
# 📌 Save Correlation Heatmap
# ===========================
plt.figure(figsize=(12, 6))
sns.heatmap(printer.corr(), annot=True, cmap='coolwarm', linewidths=1)
plt.title("Correlation Heatmap of 3D Printing Dataset")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()
