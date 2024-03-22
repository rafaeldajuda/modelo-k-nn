from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

# digits = datasets.load_digits()

### Carregue os dados de treinamento e teste
# Carregue os dados de treinamento
x_train = [[1,2,3], [31,32,33],[51,52,53]]
y_train = ["perto", "proximo","distante"]

# Carregue os dados de teste
x_test = [[21,23,24]]
y_test = ["proximo"]

# x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.5, random_state=42)
# print(type(x_train), type(x_test), type(y_train), type(y_test))

### Normalize os dados
# Normalize os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)

# Crie uma instância do classificador KNN
k = 1
knn = KNeighborsClassifier(n_neighbors=k)

# Treine o modelo:
knn.fit(x_train, y_train)

# Faça previsões com o modelo treinado
y_pred = knn.predict(x_test)
print(y_pred)

# Avalie o desempenho do modelo
accuracy = accuracy_score(y_test, y_pred)
print("Acurácia do KNN: ", accuracy * 100)

# clf = RandomForestClassifier(random_state=0)
# x = [[1,3,4], 
#      [11,12,13],
#      [21,22,23]]
# y = [0, 1, 3]
# clf.fit(x, y)
# # r = clf.predict(x)
# r = clf.predict([[4,5, 6], [14,15,16],[24,25,26]])

# print(r)

