import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Carga dataset
dataset = pd.read_csv('manufacturing.csv')

X = dataset.iloc[:, [4]].values # Material Transformation Metric
y = dataset.iloc[:, -1].values # Quality Rating

# Divide dataset en conjunto de entrenamiento 80% & prueba 20%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Entrena modelo de regresión polinómica
from sklearn.preprocessing import PolynomialFeatures
degree = 7
poly = PolynomialFeatures(degree=degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Grafica
plt.scatter(X, y, color='blue')
X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_line_poly = poly.transform(X_line)
y_line = model.predict(X_line_poly)
plt.plot(X_line, y_line, color='red')
plt.xlabel('Material Transformation Metric')
plt.ylabel('Quality Rating')
plt.title('Polynomial Regression')
plt.show()

# Evalua modelo
y_pred = model.predict(X_test_poly)

from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print("R²:", r2_score(y_test, y_pred))