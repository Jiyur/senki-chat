from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

from chat import get_response

# Khai báo ứng dụng flask
app = Flask(__name__)
cors = CORS(app, origins="*")


# [GET] http:host
@app.get("/")
def index_get():
    return render_template("index.html")


# [POST] http:host/predict
# Dự đoán nội dung người hỏi và trả về kết quả
@app.post("/predict")
def predict():
    txt = request.get_json().get("message")
    # Kiểm tra txt !=null
    res = get_response(txt)
    # JSONIFY result
    msg = {"message": res}
    return jsonify(msg)


if __name__ == "__main__":
    app.run(debug=True)
