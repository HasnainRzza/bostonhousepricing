import pickle
from flask import Flask,request, jsonify, url_for, render_template
from sklearn.preprocessing import StandardScaler
import numpy  as np
import pandas  as pd

app = Flask(__name__)

# load the model
model = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open("scaler.pkl", "rb"))

#create route
@app.route("/")
def home():
    return render_template("home.html")

#predict api
@app.route("/predict_api", methods = ['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array((list(data.values()))).reshape(1,-1))
    new_data = scaler.transform(np.array((list(data.values()))).reshape(1,-1))
    
    output = model.predict(new_data)
    print(output)
    return jsonify(output[0])


if __name__ == '__main__':
    app.run(debug=True)