from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

# Загрузите ваши данные (предположим, что data уже загружена и очищена)
data = pd.read_excel('online_retail.xlsx')

# Создание новой целевой переменной 'TotalPurchase' (общая стоимость покупки)
data['TotalPurchase'] = data['Quantity'] * data['UnitPrice']

# Удаление строк с пропущенными значениями
data = data.dropna(subset=['TotalPurchase'])

# Определяем признаки (X) и целевую переменную (y)
X = data[['Quantity', 'UnitPrice']]  # Здесь можно добавить другие признаки
y = data['TotalPurchase']

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создайте модель RandomForest
model = RandomForestRegressor(random_state=42)

# Обучение модели
model.fit(X_train, y_train)

# Прогнозирование с исходной моделью
y_pred = model.predict(X_test)

# Оценка качества исходной модели
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Random Forest - Mean Squared Error (MSE): {mse}")
print(f"Random Forest - Mean Absolute Error (MAE): {mae}")
print(f"Random Forest - R^2 Score: {r2}")

# Теперь добавим GridSearch для оптимизации гиперпараметров
# Параметры для GridSearch
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

# Обучение с использованием GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Лучшие параметры
print(f"Best parameters: {grid_search.best_params_}")

# Прогнозирование с оптимизированной моделью
best_model = grid_search.best_estimator_
y_pred_optimized = best_model.predict(X_test)

# Оценка качества оптимизированной модели
mse_optimized = mean_squared_error(y_test, y_pred_optimized)
mae_optimized = mean_absolute_error(y_test, y_pred_optimized)
r2_optimized = r2_score(y_test, y_pred_optimized)

print(f"Optimized Random Forest - Mean Squared Error (MSE): {mse_optimized}")
print(f"Optimized Random Forest - Mean Absolute Error (MAE): {mae_optimized}")
print(f"Optimized Random Forest - R^2 Score: {r2_optimized}")
