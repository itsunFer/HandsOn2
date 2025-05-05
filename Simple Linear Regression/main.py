import numpy as np

class SimpleLinearRegression:
    def __init__(self, X, Y):
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.B0 = 0
        self.B1 = 0
        self._fit()

    def _fit(self):
        n = len(self.X)
        mean_x, mean_y = np.mean(self.X), np.mean(self.Y)

        numerator = sum((self.X - mean_x) * (self.Y - mean_y))
        denominator = sum((self.X - mean_x) ** 2)

        self.B1 = numerator / denominator
        self.B0 = mean_y - self.B1 * mean_x

    def predict(self, x):
        return self.B0 + self.B1 * x

    def equation(self):
        return f"y = {self.B0:.4f} + {self.B1:.4f}x"


# Datos hardcodeados del Caso Benetton
advertising = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
sales = [5, 9, 13, 17, 20, 25, 27, 30, 35, 40]

# Crear modelo y calcular la regresión
model = SimpleLinearRegression(advertising, sales)
print("Ecuación de Regresión:", model.equation())

# Predecir valores para cinco valores desconocidos de Advertising
unknown_advertising = [15, 25, 55, 75, 95]
predictions = [model.predict(x) for x in unknown_advertising]

for x, y_pred in zip(unknown_advertising, predictions):
    print(f"Para Advertising = {x}, Predicción de Sales = {y_pred:.2f}")
