import torch
from dataset_loader import prepare_testset
from lenet import LeNet


def load_checkpoint(filepath):
    torch.no_grad()
    checkpoint = torch.load(filepath)
    model = LeNet()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def make_inference(modelpath, test_set):
    batch_data, classes = prepare_testset(test_set)
    inference_model = load_checkpoint(modelpath)
    outputs = inference_model(batch_data)
    _, prediction = torch.max(outputs.data, 1)
    for i in range(batch_data.__len__()):
        print("Predicted: ", classes[prediction[i]])
        print(prediction, classes[prediction[i]])
    return prediction
