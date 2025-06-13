import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
import xgboost as xgb

# ✅ Load CSV (update if needed)
df = pd.read_csv('/Users/amiralibozorgnia/Downloads/parameters.csv')  # <-- Update path

# ✅ Encode 'Orientation' (F=0, V=1, H=2)
df['Orientation'] = df['Orientation'].map({'F': 0, 'V': 1, 'H': 2})

# ✅ Define features and target
X = df.drop(columns=['sample', 'UTS'], errors='ignore')
y = df['UTS']

# ✅ Convert all to numeric (in case of string-type numbers)
X = X.apply(pd.to_numeric, errors='coerce')

# ✅ Add interaction terms
X['inter_OR_Nozzle'] = X['Orientation'] * X['Nozzle Temp.']
X['inter_OR_Bed'] = X['Orientation'] * X['Bed Temp']
X['inter_Bed_Nozzle'] = X['Bed Temp'] * X['Nozzle Temp.']
X['inter_Flow_Speed'] = X['Flow Rate'] * X['Printing Speed']

# ✅ Drop any rows with NaNs
X = X.dropna()
y = y.loc[X.index]  # align y with cleaned X

# ✅ Standardize
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# ✅ Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ✅ Models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=150, max_depth=6, random_state=42),
    'Decision Tree': DecisionTreeRegressor(max_depth=3, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4,
                                subsample=0.9, colsample_bytree=0.9, random_state=42)
}

# ✅ Train and evaluate
r2_results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    r2_results[name] = r2
    print(f"{name} R²: {r2:.3f}")

# ✅ Plot
r2_df = pd.DataFrame.from_dict(r2_results, orient='index', columns=['R2 Score'])

plt.figure(figsize=(8, 5))
sns.barplot(x=r2_df.index, y=r2_df['R2 Score'])
plt.title("Model R2 Score Comparison (UTS Prediction)")
plt.xticks(rotation=30)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("uts_r2_comparison.png", dpi=300, bbox_inches='tight')
plt.show()
