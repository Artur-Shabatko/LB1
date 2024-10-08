# Імпорт необхідних бібліотек
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# 1. Генерація випадкових даних
data = np.random.randint(1, 1001, 1000).reshape(-1, 1)  # 1000 випадкових значень від 1 до 1000

# 2. Нормалізація даних
scaler = MinMaxScaler()  # Ініціалізація Min-Max нормалізатора
norm_data = scaler.fit_transform(data)  # Нормалізація даних

# Генерація міток (для регресії беремо випадкові значення)
labels = np.random.randint(1, 1001, 1000)

# 3. Розділення на навчальну і тестову вибірки (70% для навчання, 30% для тестування)
X_train, X_test, y_train, y_test = train_test_split(norm_data, labels, test_size=0.3, random_state=42)

# 4. Навчання KNN-регресора для різних K
knn_1 = KNeighborsRegressor(n_neighbors=1)
knn_3 = KNeighborsRegressor(n_neighbors=3)
knn_5 = KNeighborsRegressor(n_neighbors=5)

knn_1.fit(X_train, y_train)  # Навчання для K=1
knn_3.fit(X_train, y_train)  # Навчання для K=3
knn_5.fit(X_train, y_train)  # Навчання для K=5

# Прогнозування на тестових даних для кожного K
pred_1 = knn_1.predict(X_test)
pred_3 = knn_3.predict(X_test)
pred_5 = knn_5.predict(X_test)

# 5. Оцінка якості регресії за допомогою середньоквадратичної помилки (MSE)
mse_1 = mean_squared_error(y_test, pred_1)
mse_3 = mean_squared_error(y_test, pred_3)
mse_5 = mean_squared_error(y_test, pred_5)

print(f'MSE для K=1: {mse_1}')
print(f'MSE для K=3: {mse_3}')
print(f'MSE для K=5: {mse_5}')

# 6. Візуалізація результатів
plt.figure(figsize=(10, 6))
plt.plot(range(len(y_test)), y_test, color='blue', label='Тестові дані')
plt.plot(range(len(pred_3)), pred_3, color='red', label='Прогнозовані значення (K=3)')
plt.title('Порівняння тестових даних та прогнозованих значень для K=3')
plt.xlabel('Індекс')
plt.ylabel('Значення')
plt.legend()
plt.show()
