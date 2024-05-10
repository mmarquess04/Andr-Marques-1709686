import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import neighbors
import pandas as pd

# Load optdigits dataset
data = pd.read_csv("Dataset/optdigits.tes", sep=",", header=None)

# Extract features and target
X = data.iloc[:, :-1]  # Features are all columns except the last one
y = data.iloc[:, -1]   # Target is the last column

# Apply PCA for dimensionality reduction to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Define number of neighbors
n_neighbors = 15

# Create meshgrid for plotting decision boundaries
h = 0.2  # step size in the mesh
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Plot decision boundaries for uniform weight type
clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
clf.fit(X_pca, y)

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

# Plot data points
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolor='k')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Optdigits classification (k = %i, weights = 'uniform')" % n_neighbors)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.tight_layout()
plt.show()