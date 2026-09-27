import numpy as np
import pandas as pd
# 1. NumPy Array Operations
marks = np.array([72, 85, 91, 68, 77])
print("Marks:", marks)
print("Mean:", np.mean(marks))
print("Maximum:", np.max(marks))
print("Minimum:", np.min(marks))
# 2. Pandas DataFrame Creation
data = {
"Name": ["Amit", "Riya", "Sourav", "Neha", "Rahul"],
"Attendance": [88, 92, 76, 95, 81],
"Marks": [72, 85, 68, 91, 77]
}
df = pd.DataFrame(data)
# 3. Data Exploration
print("\n--- First Five Records ---")
print(df.head())
print("\n--- Data Information ---")
print(df.info())
print("\n--- Statistical Summary ---")
print(df.describe())
print("\nAverage Marks:", df["Marks"].mean())
