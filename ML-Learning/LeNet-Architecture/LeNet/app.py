from flask import jsonify
from src.routes import app


@app.get('/')
def home():

    return jsonify(f'Hello World!, this is an endpoint for introducing the LeNet model in PyTorch.')


if __name__ == "__main__":
    app.run(debug=True)
