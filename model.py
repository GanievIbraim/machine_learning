import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import joblib

# Загрузка данных
data = pd.read_excel('online_retail.xlsx')

# Удаление строк с пропущенными значениями в CustomerID
data = data.dropna(subset=['CustomerID'])

# Преобразование InvoiceDate в datetime
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

# Расчёт TotalSpend (общая сумма покупок)
data['TotalSpend'] = data['Quantity'] * data['UnitPrice']

# Группировка данных по клиентам
clv_data = data.groupby('CustomerID').agg({
    'TotalSpend': 'sum',  # Общая сумма покупок
    'InvoiceNo': 'nunique',  # Количество покупок (Frequency)
    'InvoiceDate': lambda x: (data['InvoiceDate'].max() - x.max()).days,  # Recency (время с последней покупки)
}).reset_index()

# Переименование столбцов
clv_data.rename(columns={
    'InvoiceNo': 'Frequency',
    'InvoiceDate': 'Recency'
}, inplace=True)

# Расчёт AverageSpend (средняя сумма покупки)
clv_data['AverageSpend'] = clv_data['TotalSpend'] / clv_data['Frequency']

# Удаление выбросов с помощью Z-score
clv_data = clv_data[(np.abs(stats.zscore(clv_data[['TotalSpend', 'Frequency', 'Recency', 'AverageSpend']])) < 3).all(axis=1)]

# Определение признаков (X) и целевой переменной (y)
X = clv_data.drop(['CustomerID', 'TotalSpend'], axis=1)
y = clv_data['TotalSpend']  # Целевая переменная: TotalSpend (CLV)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели RandomForest с настройкой гиперпараметров
model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
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

# Сохранение модели в файл
joblib.dump(model, 'clv_model_no_country.pkl')