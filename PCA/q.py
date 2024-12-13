import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

mnist = fetch_openml('mnist_784')
X, y = mnist["data"], mnist["target"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


def pca(X, n_components=2):
    X_centered = X - np.mean(X, axis=0)
    covariance_matrix = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_idx]
    eigenvectors = eigenvectors[:, sorted_idx]
    eigenvectors = eigenvectors[:, :n_components]
    X_pca = np.dot(X_centered, eigenvectors)
    return X_pca

X_pca = pca(X_scaled, 2)
plt.figure(figsize=(8, 6))
for i in range(10):
  plt.scatter(X_pca[y == str(i), 0], X_pca[y == str(i), 1], label = str(i))
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Traditional PCA')
plt.legend()
plt.savefig('pca_q5.png')


tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)
plt.figure(figsize=(8, 6))
for i in range(10):
  plt.scatter(X_tsne[y == str(i), 0], X_tsne[y == str(i), 1], label = str(i))
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Traditional PCA')
plt.legend()
plt.savefig('tsne_q5.png')


# PCA is a linear dimensionality reduction technique that preserves the global structure of the data, but it may not capture complex, non-linear relationships.
# t-SNE, on the other hand, is a non-linear dimensionality reduction technique that is particularly good at preserving local structure and revealing clusters in the data.
# In terms of discriminability of samples, t-SNE often provides a better separation of clusters compared to PCA, making it easier to distinguish between different classes.