from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump

# Carrega o conjunto de dados Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Divide os dados em conjuntos de treinamento e teste (2/3 para treinamento, 1/3 para teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42)

# Redução de dimensão usando PCA (2 componentes)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)

# Cria o classificador de regressão logística
logreg = LogisticRegression(max_iter=1000)

# Treina o classificador com os dados reduzidos
logreg.fit(X_train_pca, y_train)

# Reduz a dimensão dos dados de teste
X_test_pca = pca.transform(X_test)

# Faz previsões
y_pred = logreg.predict(X_test_pca)

# Salva o modelo treinado, o x_test_pca e o y_test
dump(logreg, 'logistic_regression_model.joblib')
dump(X_test_pca, 'X_test_pca.joblib')
dump(y_test, 'y_test.joblib')
dump(y, 'y.joblib')
dump(iris, 'iris.joblib')

# Avalia a precisão do modelo
accuracy = accuracy_score(y_test, y_pred)
print("Precisão do modelo Logistic Regression com PCA:", accuracy)