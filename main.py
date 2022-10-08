from flask import Flask, request, jsonify
import xgboost as xgb
import pandas as pd
from flask_cors import CORS




app = Flask(__name__)

CORS(app)

model = xgb.XGBClassifier()
model.load_model("model.json")


@app.route("/test", methods = ["POST"])
def predict():
   json_data = request.json
   df = pd.DataFrame([json_data])
   prediction = model.predict(df)
   pred = prediction.tolist()
   return jsonify({"prediction": pred[0]})