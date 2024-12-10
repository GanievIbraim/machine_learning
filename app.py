from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS  # Импортируем библиотеку для работы с CORS

app = Flask(__name__)
CORS(app)  # Разрешаем CORS для всего приложения

# Загрузка модели
model = joblib.load('clv_model_no_country.pkl')

# Получение списка признаков, использованных при обучении модели
model_features = model.feature_names_in_

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        # Ответ на предварительный запрос OPTIONS
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response

    try:
        # Логирование запроса
        print("Received request:", request.json)

        # Получение данных от клиента
        data = request.json

        # Проверка входных данных
        if not isinstance(data, dict) or not all(key in data for key in ['Frequency', 'Recency', 'AverageSpend']):
            return jsonify({'error': 'Invalid input format. Expected {"Frequency": ..., "Recency": ..., "AverageSpend": ...}'}), 400

        # Преобразование данных в DataFrame
        new_customer_data = pd.DataFrame(data, index=[0])

        # Проверка типов данных
        if not all(isinstance(value, (int, float)) for value in [data['Frequency'], data['Recency'], data['AverageSpend']]):
            return jsonify({'error': 'Frequency, Recency, and AverageSpend must be numbers'}), 400

        # Убедимся, что порядок признаков совпадает с порядком, использованным при обучении
        new_customer_data = new_customer_data[model_features]

        # Прогнозирование
        prediction = model.predict(new_customer_data)

        # Логирование предсказания
        print("Prediction:", prediction)

        # Возвращение результата
        response = jsonify({'prediction': prediction.tolist()})
        response.headers.add('Access-Control-Allow-Origin', '*')  # Разрешаем CORS
        return response
    except Exception as e:
        # Логирование ошибки
        print("Error:", e)
        response = jsonify({'error': str(e)})
        response.headers.add('Access-Control-Allow-Origin', '*')  # Разрешаем CORS
        return response, 500

if __name__ == '__main__':
    app.run(debug=True)