import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Загрузка данных
data = pd.read_excel('online_retail.xlsx')

# Создание новой целевой переменной 'TotalPurchase' (общая стоимость покупки)
data['TotalPurchase'] = data['Quantity'] * data['UnitPrice']

# Удаление строк с пропущенными значениями
data = data.dropna(subset=['TotalPurchase'])

# Определяем признаки (X) и целевую переменную (y)
X = data[['Quantity', 'UnitPrice']]  # Здесь можно добавить другие признаки
y = data['TotalPurchase']

# Разделение данных на обучающую (60%), валидационную (20%) и тестовую (20%) выборки
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Обучение модели RandomForest
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Прогнозирование на тестовых данных
y_pred = model.predict(X_test)

# Оценка качества модели
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Random Forest - Mean Squared Error (MSE): {mse}")
print(f"Random Forest - Mean Absolute Error (MAE): {mae}")
print(f"Random Forest - R^2 Score: {r2}")

# Прогнозирование для нового клиента (предположим, что данные нового клиента будут такими же)
new_customer_data = pd.DataFrame({
    'Quantity': [10],  # Пример данных для нового клиента
    'UnitPrice': [5.0]
})

new_customer_purchase = model.predict(new_customer_data)
print(f"Predicted Total Purchase for new customer: {new_customer_purchase[0]}")