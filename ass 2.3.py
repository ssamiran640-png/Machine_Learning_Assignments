import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)

plt.figure(figsize=(12, 10))
corr_matrix = df.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap of Wine Dataset Features")
plt.tight_layout()
plt.show()

corr_pairs = corr_matrix.unstack()
corr_pairs = corr_pairs[corr_pairs < 1.0] 
strongest_positive = corr_pairs.idxmax()
print("Pair with strongest positive correlation:", strongest_positive)
print("Correlation value:", corr_pairs.max())