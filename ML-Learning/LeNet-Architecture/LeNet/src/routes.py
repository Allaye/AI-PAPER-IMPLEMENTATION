from flask import Flask, jsonify
from . import inference

app = Flask(__name__)


@app.get('/')
def predict():
    d = inferance
    print(d)
    return jsonify(f'Hello World!, this is an endpoint for introducing the LeNet model in PyTorch.')
