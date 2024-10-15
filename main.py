import numpy as np
import matplotlib.pyplot as plt

# 1. Генерація випадкових даних (більш складний розподіл)
np.random.seed(1)
X = np.random.rand(1000, 1) * 10  # 1000 випадкових точок у діапазоні [0, 10]
# Цільова змінна є комбінацією синуса, косинуса і випадкового шуму
# Генерація більш випадкової цільової змінної y
y = (
        np.sin(X).ravel()  # синус
        + 0.5 * np.cos(2 * X).ravel()  # косинус
        + 0.3 * np.random.rand(1000)  # випадковий шум
        + 0.2 * X.ravel() * np.random.normal(0, 0.2, 1000)  # залежність від X із шумом
        + np.random.normal(0, 0.5, 1000)  # ще більше шуму
)

# Виведення початкової вибірки без нормалізації
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', s=10, label='Початкові дані')
plt.title('Початкова вибірка без нормалізації')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# 2. Нормалізація даних (мінімум-максимум)
X_norm = (X - X.min()) / (X.max() - X.min())

# Виведення графіка нормалізованих даних
plt.figure(figsize=(8, 6))
plt.scatter(X_norm, y, color='green', s=10, label='Нормалізовані дані')
plt.title('Нормалізована вибірка')
plt.xlabel('Нормалізоване X')
plt.ylabel('y')
plt.legend()
plt.show()

# 3. Розподіл на навчальну та тестову вибірки (80% навчання, 20% тест)
split_idx = int(0.8 * len(X_norm))
X_train, X_test = X_norm[:split_idx], X_norm[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Виведення графіка після розподілу даних на тренувальну та тестову вибірки
plt.figure(figsize=(10, 6))

# Тренувальна вибірка
plt.scatter(X_train, y_train, color='blue', s=10, label='Тренувальна вибірка')

# Тестова вибірка
plt.scatter(X_test, y_test, color='orange', s=10, label='Тестова вибірка')

plt.title('Тренувальна та тестова вибірки')
plt.xlabel('Нормалізоване X')
plt.ylabel('y')
plt.legend()
plt.show()


# 4. Реалізація KNN-регресії
def knn_regressor(X_train, y_train, X_test, k):
    y_pred = []
    for x_test in X_test:
        # Обчислення відстані від тестової точки до всіх точок тренувальної вибірки
        distances = np.sqrt(np.sum((X_train - x_test) ** 2, axis=1))
        # Сортуємо відстані і вибираємо K найближчих сусідів
        k_indices = distances.argsort()[:k]
        # Усереднюємо значення сусідів
        y_pred.append(np.mean(y_train[k_indices]))
    return np.array(y_pred)


# 5. Пошук оптимального K з покроковим виводом
k_values = range(1, 11)  # розглядаємо K від 1 до 20
errors = []

for k in k_values:
    print(f"\n=== Обчислення для K = {k} ===")
    y_pred = knn_regressor(X_train, y_train, X_test, k)
    mse = np.mean((y_test - y_pred) ** 2)  # середньоквадратична похибка (MSE)
    print(f"MSE для K = {k}: {mse}")
    errors.append(mse)

    # Покрокова візуалізація результатів
    plt.figure(figsize=(6, 4))
    plt.scatter(X_train, y_train, label='Тренувальні Дані', color='blue', s=5)
    plt.scatter(X_test, y_test, label='Тестові дані', color='yellow', s=10)
    plt.scatter(X_test, y_pred, label=f'Прогноз (K = {k})', color='red', s=10)
    plt.title(f'Прогноз для K = {k}')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()

best_k = k_values[np.argmin(errors)]
print(f"\nНайкраще значення K: {best_k}")

# 6. Візуалізація MSE для різних K
plt.figure(figsize=(10, 6))
plt.plot(k_values, errors, marker='o')
plt.title('Помилка (MSE) на тестовій вибірці для різних K')
plt.xlabel('K')
plt.ylabel('MSE')
plt.show()