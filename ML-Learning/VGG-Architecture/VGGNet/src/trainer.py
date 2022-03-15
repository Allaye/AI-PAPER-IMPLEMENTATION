import torch
from torchvision.transforms import transforms

from vggnet import VGGNet
from dataloader import CustomDataset
from setting import *


def training(model, dataloader, hyperparameters, device):
    epochs, lr, batch_size, momentum = hyperparameters
    loss_fn, optimizer = model.loss_optimizer(lr, momentum)
    training_loss = 0.0
    last_loss = 0.0
    for epoch in range(epochs):
        model.train(True)
        print('training')
        for index, (images, label) in enumerate(dataloader[0]):  # image b
            # move image operation to cuda
            images = images.to(device)
            label = label.to(device)
            # clear the gradient values and perform a backward and optimization step
            optimizer.zero_grad()
            # perform a forward pass
            outputs = model(images).to(device)
            # compute the loss values, perform a backward pass and update the weights/optimizer and record the loss value
            loss = loss_fn(outputs, label)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
            # print training statistics and other information
            if index % 100 == 99:
                print(
                    f'Training Info: Epoch [{epoch + 1}/{epochs}], Step [{index + 1}/{len(dataloader)}] Avg_training_loss: {training_loss / 100:.4f}')
        eval_model(model, dataloader[1], device, loss_fn)


def eval_model(model, test_loader, device, loss_fn=None):
    model.eval()
    correct = 0
    total_loss = 0
    validation_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            validation_loss += loss.item()
        avg_val_loss = validation_loss / len(test_loader)
        print(f'Validation Info: Avg_validation_loss: {avg_val_loss:.4f}')


def save_model(model, epoch):
    pass


from setting import config_device, hyperparameter
from dataloader import CustomDataset
from architecture import config

d_path = "C:\Python\Project\Personal\Python Project\Projects\Data\Algo-ML\dataset"

data = CustomDataset(d_path, transform=[transforms.ToTensor(), transforms.Resize((224, 224))])
dataset = data.getdataloader()

# training(m, trainset, hyper, device)
torch.cuda.empty_cache()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(dev)
vgg = VGGNet(config['vgg16-C1']).to(dev)
# vgg.train(True)
# x = torch.randn(1, 3, 224, 224).to(device)
hyper = hyperparameter()

training(vgg, dataset, hyper, dev)
