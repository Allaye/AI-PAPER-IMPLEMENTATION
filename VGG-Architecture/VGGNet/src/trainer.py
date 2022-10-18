import torch
from torchvision.transforms import transforms
from setting import config_device, hyperparameter
from dataloader import CustomDataset
from architecture import config
from vggnet import VGGNet
from dataloader import CustomDataset
from setting import *


def training(model: VGGNet, dataloader, hyperparameters, device):
    epochs, lr, batch_size, momentum = hyperparameters
    loss_fn, optimizer = model.loss_optimizer(lr, momentum)
    training_loss = 0.0
    last_loss = 0.0
    for epoch in range(epochs):
        model.train(True)
        print(f'training.&.validation..............................epoch....{epoch + 1} of {epochs}')
        for index, (images, label) in enumerate(dataloader[0]):  # image b
            # move image operation to cuda
            images = images.to(device)
            label = label.to(device)
            # clear the gradient values and perform a backward and optimization step
            optimizer.zero_grad()
            # perform a forward pass
            outputs = model(images).to(device)
            # compute the loss values, perform a backward pass and update the weights/optimizer and record the loss
            # value
            loss = loss_fn(outputs, label)
            loss.backward()
            optimizer.step()
            training_loss += loss.item() * images.size(0)
            # print training statistics and other information
            if index % 100 == 99:
                print(
                    f'Training Info: Epoch [{epoch + 1}/{epochs}], Step [{index + 1}/{len(dataloader[0])}] Avg_training_loss: {loss.item():.4f}')
        training_loss = training_loss / len(dataloader[0])
        print(f'Training Info after epoch {epoch + 1} of {epochs}: Avg_training_loss: {training_loss:.4f}')
        eval_model(model, dataloader[1], device, loss_fn)


def eval_model(model, test_loader, device, loss_fn=None):
    model.eval()
    correct = 0
    total_loss = 0
    validation_loss = 0.0
    total_sample = 0
    total_correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).to(device)
            loss = loss_fn(outputs, labels)
            validation_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_sample += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            # total_correct += predicted.eq(labels).sum().item()
            # total_correct = total_correct + (predicted == labels).sum().item()

        avg_val_loss = validation_loss / len(test_loader)
        print(f'Validation Info: Avg_validation_loss: {avg_val_loss:.4f} Validation_loss: {validation_loss:.4f}')


def save_model(model, epoch, optimizer, best_acc):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc
    }
    torch.save(checkpoint, 'checkpoint.pth')


if __name__ == '__main__':
    d_path = "./dataset"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = CustomDataset(d_path, transform=[transforms.ToTensor(), transforms.Resize((224, 224))])
    dataset = data.getdataloader()
    vgg = VGGNet(config['vgg16-C1']).to(dev)
    hyper = hyperparameter()
    training(vgg, dataset, hyper, dev)
