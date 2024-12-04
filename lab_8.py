import pandas as pd

# Шаг 1. Загрузка данных
file_path = 'data.csv'

# Пропуск первых строк, содержащих лишние заголовки или пустые строки
data = pd.read_csv(file_path, skiprows=4)

# Шаг 2. Присвоение корректных названий столбцов
data.columns = ['Account Customer', 'FR', 'DFL', 'TP', 'DLG', 'Total']

# Шаг 3. Удаление пустых строк
data = data.dropna(how='all').reset_index(drop=True)

# Шаг 4. Преобразование числовых данных
numeric_columns = ['FR', 'DFL', 'TP', 'DLG', 'Total']
data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Шаг 5. Удаление строк с пропущенными значениями в числовых колонках
data = data.dropna(subset=numeric_columns, how='any').reset_index(drop=True)

# Шаг 6. Вывод итоговых данных
print("Обработанные данные:")
print(data)

# Сохранение в новый файл
data.to_csv('cleaned_data.csv', index=False)
