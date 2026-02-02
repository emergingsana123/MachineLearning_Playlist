import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.datasets import make_classification

# Generate synthetic binary classification data
X, y = make_classification(
    n_samples=500,
    n_features=2,        # keep 2 for easy visualization
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=1.5,
    random_state=42
)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', alpha=0.7)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Synthetic Binary Classification Data")
plt.show()
