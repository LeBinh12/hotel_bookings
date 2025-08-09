# file: api.py
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import tensorflow as tf

# Load model và preprocessor
model = tf.keras.models.load_model('hotel_model.h5')
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # nhận dữ liệu JSON từ PHP

    # Chuyển thành DataFrame
    df = pd.DataFrame([data])

    # Xử lý dữ liệu
    X = preprocessor.transform(df)

    # Dự đoán xác suất
    prob = model.predict(X)[0][0]
    result = int(prob >= 0.5)  # 1: hủy, 0: không hủy

    return jsonify({
        'probability': float(prob),
        'prediction': result
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
