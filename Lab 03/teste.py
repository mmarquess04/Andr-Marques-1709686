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

# Carregar o modelo K-means treinado e os dados transformados
with open('kmeans_model.pkl', 'rb') as f:
    kmeans = pickle.load(f)

with open('X_train_pca.pkl', 'rb') as f:
    X_train_pca = pickle.load(f)

# Aplicar PCA aos dados de teste
pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test)

# Usar o modelo K-means para prever os clusters dos dados de teste
y_pred_test = kmeans.predict(X_test_pca)

# Salvar os dados de teste transformados e as previsões
with open('X_test_pca.pkl', 'wb') as f:
    pickle.dump(X_test_pca, f)

with open('y_pred_test.pkl', 'wb') as f:
    pickle.dump(y_pred_test, f)

# Plotar os dados de teste e os centróides
plt.figure(figsize=(8, 6))
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_pred_test, cmap='viridis', marker='o', s=90, edgecolor='face')
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker="o", s=90, linewidths=1, color="red", edgecolor="black")
plt.title("Distribuição de bolhas - Dados de Teste")
plt.xlabel("Componente principal 1")
plt.ylabel("Componente principal 2")
plt.show()






