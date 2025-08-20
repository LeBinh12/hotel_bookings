from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load("hotel_cancellation_model.joblib")

# Trang chủ với form nhập dữ liệu
@app.route('/')
def home():
    return render_template('index.html')

# Xử lý dự đoán
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Lấy dữ liệu từ form
        features = {
            "lead_time": int(request.form['lead_time']),
            "adr": float(request.form['adr']),
            "stays_in_weekend_nights": int(request.form['stays_in_weekend_nights']),
            "stays_in_week_nights": int(request.form['stays_in_week_nights']),
            "adults": int(request.form['adults']),
            "children": int(request.form['children']),
            "babies": int(request.form['babies']),
            "meal": request.form['meal'],
            "market_segment": request.form['market_segment'],
            "distribution_channel": request.form['distribution_channel'],
            "reserved_room_type": request.form['reserved_room_type'],
            "deposit_type": request.form['deposit_type'],
            "customer_type": request.form['customer_type'],
            "hotel": request.form['hotel'],
            "required_car_parking_spaces": int(request.form['required_car_parking_spaces']),
            "total_of_special_requests": int(request.form['total_of_special_requests'])
        }


        # Convert sang DataFrame để model xử lý
        input_df = pd.DataFrame([features])

        # Dự đoán
        prob = model.predict_proba(input_df)[:, 1][0] * 100
        prediction = "Khách hàng CÓ THỂ hủy đặt phòng" if prob > 50 else "Khách hàng ÍT KHẢ NĂNG hủy đặt phòng"

        return render_template('result.html', prediction=prediction, probability=round(prob, 2))

if __name__ == '__main__':
    app.run(debug=True)
