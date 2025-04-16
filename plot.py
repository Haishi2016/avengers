import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg', depending on your system


# Replace this with your actual output path
output_dir = "/home/hbai/projects/go/src/github.com/Haishi2016/avengers/output"

# Read all part files into one DataFrame
files = glob.glob(f"{output_dir}/part-*.csv")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

y_test = df["label"]
y_pred = df["prediction"]

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.xlabel("True Bed Elevation")
plt.ylabel("Predicted Bed Elevation")
plt.title("XGBoost Predictions vs. Ground Truth")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.grid(True)
plt.tight_layout()
plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")
print(f"R²: {r2:.3f}")
