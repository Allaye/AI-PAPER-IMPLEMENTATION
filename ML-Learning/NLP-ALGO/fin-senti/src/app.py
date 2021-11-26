from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import torch
import model
import configuration
import inferance

app = Flask(__name__)

MODEL = None


@app.get("/predict")
def predict():
    text = request.args.get("text")
    positive_inferance = inferance.premise_inferance(text, MODEL)
    negative_inferance = 1 - positive_inferance
    response = {}
    response["inferance"] = {
        "positive": str(positive_inferance),
        "negative": str(negative_inferance),
        "text": text
    }
    return jsonify(response)


if __name__ == "__main__":
    MODEL = model.BertUncanned()
    MODEL.load_state_dict(torch.load(configuration.MODEL_PATH))
    MODEL.to(configuration.DEVICE)
    MODEL.eval()
    app.run(debug=True, port=8080)
