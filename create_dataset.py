import pandas as pd
import random

# Функция для генерации случайных данных
def generate_data(num_records=500000):
    data = {
        "transaction_amount": [random.randint(10, 1000) for _ in range(num_records)],
        "product_category": [random.choice(["electronics", "clothing", "home", "beauty", "sports"]) for _ in range(num_records)],
        "quantity": [random.randint(1, 10) for _ in range(num_records)],
        "age": [random.randint(18, 70) for _ in range(num_records)],
        "gender": [random.choice(["M", "F"]) for _ in range(num_records)],
        "location": [random.choice(["NY", "CA", "TX", "FL", "IL"]) for _ in range(num_records)],
        "purchase_frequency": [random.randint(1, 20) for _ in range(num_records)],
        "average_order_value": [random.randint(50, 500) for _ in range(num_records)],
        "returns_count": [random.randint(0, 5) for _ in range(num_records)],
        "CLV": [random.randint(100, 10000) for _ in range(num_records)],
    }
    return pd.DataFrame(data)

# Генерация датасета
df = generate_data(500000)

# Сохранение в CSV
df.to_csv("dataset.csv", index=False)

print("Датасет успешно создан и сохранен в файл dataset.csv")
