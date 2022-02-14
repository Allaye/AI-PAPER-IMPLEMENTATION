from flask import Flask, jsonify

# from LeNet.src import inferance
from src import inferance

app = Flask(__name__)


@app.get('/')
def home():
    d = inferance
    print(d)
    return jsonify(f'Hello World!, this is an endpoint for introducing the LeNet model in PyTorch.')


if __name__ == "__main__":
    app.run(debug=True)
