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
    print("iam output",output)
    return jsonify(output[0])

@app.route("/predict", methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output = model.predict(final_input)[0]
    return render_template("home.html", prediction_text = "this house is priced at {}".format(output))

if __name__ == '__main__':
    app.run(debug=True)