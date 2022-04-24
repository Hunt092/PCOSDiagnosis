from black import out
from flask import Flask, jsonify, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("bigboymodel.pkl", "rb"))


def seletect(x):
    try:
        y = int(x)
        return y
    except:
        try:
            y = float(x)
            return y
        except:
            return x


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")
    # Get the data from the POST request.
    data = [x for x in request.form.values()]
    numarray = np.array(data).reshape(1, -1)
    # Make prediction using model loaded from disk as per the data.
    answer = model.predict(numarray)

    if answer[0]:
        output = True
    else:
        output = False
    return render_template("index.html", output=output)


if __name__ == "__main__":
    app.run(port=5000, debug=True)
