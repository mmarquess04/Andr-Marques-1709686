import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA  # Importa PCA
from sklearn import datasets

# Carrega o conjunto de dados Iris
iris = datasets.load_iris()
X = iris.data  # Usando todos os atributos
y = iris.target

# Redução de dimensionalidade usando PCA
pca = PCA(n_components=2)  # Reduzindo para 2 componentes
X_pca = pca.fit_transform(X)

h = 0.02  # Tamanho do passo na malha

# Cria mapas de cores
cmap_light = ListedColormap(["orange", "cyan", "cornflowerblue"])
cmap_bold = ["darkorange", "c", "darkblue"]

# Cria o classificador de regressão logística
logreg = LogisticRegression(max_iter=1000)  # Você pode ajustar max_iter se necessário

# Treina o classificador com os dados reduzidos
logreg.fit(X_pca, y)

# Define os limites do gráfico
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1

# Cria a malha
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Faz previsões para os rótulos de cada ponto na malha
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

# Coloca o resultado em um gráfico de cores
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light)

# Plota os pontos de treinamento
sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=iris.target_names[y],
    palette=cmap_bold,
    alpha=1.0,
    edgecolor="black",
)

# Define os limites e rótulos do gráfico
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Classificação em 3 classes (Regressão Logística com PCA)")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")

# Mostra o gráfico
plt.show()