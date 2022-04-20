from flask import Flask, jsonify, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")
    # Get the data from the POST request.
    data = [x for x in request.form.values()]

    # Make prediction using model loaded from disk as per the data.
    print(data)
    return render_template("index.html")


if __name__ == "__main__":
    app.run(port=5000, debug=True)
