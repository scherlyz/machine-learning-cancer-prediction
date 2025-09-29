import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA

# ========== PERSIAPAN DATA ==========
df = pd.read_csv("risk_factors_cervical_cancer.csv")

# Ganti '?' dan kosong jadi NaN
df.replace(['?', ' ', ''], np.nan, inplace=True)

# Konversi ke numerik
df = df.apply(pd.to_numeric, errors='coerce')

# Isi NaN dengan median
df.fillna(df.median(), inplace=True)

target_col = 'Biopsy'

# Encode target jika perlu
le = LabelEncoder()
df[target_col] = le.fit_transform(df[target_col])

X = df.drop(target_col, axis=1)
y = df[target_col]

# Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Hilangkan fitur constant
var_thresh = VarianceThreshold(threshold=0.0)
X_var = var_thresh.fit_transform(X_scaled)

# Feature selection ANOVA
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X_var, y)

# Balancing dengan SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_selected, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=1, stratify=y_res
)

# ========== VISUALISASI DISTRIBUSI ==========
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Distribusi sebelum SMOTE
pd.Series(y).value_counts().sort_index().plot(kind='bar', ax=axes[0], color=['skyblue', 'salmon'])
axes[0].set_title('Distribusi Kelas Sebelum SMOTE')
axes[0].set_xlabel('Kelas')
axes[0].set_ylabel('Jumlah Sampel')

# Distribusi setelah SMOTE
pd.Series(y_res).value_counts().sort_index().plot(kind='bar', ax=axes[1], color=['skyblue', 'salmon'])
axes[1].set_title('Distribusi Kelas Setelah SMOTE')
axes[1].set_xlabel('Kelas')
axes[1].set_ylabel('Jumlah Sampel')

plt.tight_layout()
plt.show()

# ========== VISUALISASI DATA TRAIN ==========
# PCA untuk reduksi ke 2D
pca = PCA(n_components=2, random_state=42)
X_train_2d = pca.fit_transform(X_train)

plt.figure(figsize=(6, 5))
plt.scatter(X_train_2d[y_train == 0, 0], X_train_2d[y_train == 0, 1],
            alpha=0.6, label='Kelas 0', c='skyblue', edgecolor='k')
plt.scatter(X_train_2d[y_train == 1, 0], X_train_2d[y_train == 1, 1],
            alpha=0.6, label='Kelas 1', c='salmon', edgecolor='k')

plt.title("Visualisasi Data Train (PCA 2D)")
plt.xlabel("Komponen PCA 1")
plt.ylabel("Komponen PCA 2")
plt.legend()
plt.tight_layout()
plt.show()
