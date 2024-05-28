import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pickle

# Atividade 1: Carregar o dataset digits
X, y = load_digits(return_X_y=True)

# Atividade 4: Estratificação do dataset em dados de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.33, random_state=42)

# Atividade 3: Redução de dimensionalidade para 2D com PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)

# Atividade 2: Aplicar K-means ao dataset transformado
kmeans = KMeans(init="k-means++", n_clusters=10, n_init=10, random_state=42)
y_pred = kmeans.fit_predict(X_train_pca)

# Salvar o modelo K-means e os dados transformados
with open('kmeans_model.pkl', 'wb') as f:
    pickle.dump(kmeans, f)

with open('X_train_pca.pkl', 'wb') as f:
    pickle.dump(X_train_pca, f)

# Plotar os dados de treino e os centróides
plt.figure(figsize=(8, 6))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_pred, cmap='viridis', marker='o', s=90, edgecolor='face')
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker="o", s=90, linewidths=1, color="red", edgecolor="black")
plt.title("Distribuição anisotrópica de bolhas - Dados de Treino")
plt.xlabel("Componente principal 1")
plt.ylabel("Componente principal 2")
plt.show()




