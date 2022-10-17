import torch
from lenet import LeNet
from dataset_loader import prepare_dataset
from torch.utils.data import Dataset, DataLoader
from configuration import hyperparameter, configure_device
from utils import save_checkpoint


def train(model: LeNet, train_loader: DataLoader, test_loader: DataLoader, epochs: int, loss_fn: torch.nn.modules.loss, device, batch_size, optimizer) -> torch.nn.Module:
    """
    perform model training loop and hyperparameter tuning
    :param model:
    :param train_loader:
    :param test_loader:
    :param epochs:
    :param loss_fn:
    :param device:
    :param batch_size:
    :param optimizer:
    :rtype: model trained and tuned
    """
    best_accuracy = 0
    n_total_steps = len(train_loader)
    print(f"starting training now!")
    for epoch in range(epochs):
        # put model in training mode and engage it to track the gradients
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            # move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)
            # perform a forward pass
            outputs = model(images)

            # calculate the loss (error rate or error margin)
            loss = loss_fn(outputs, labels)

            # engage model gradients and perform a backward and optimize step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print training statistics and other information
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')
            # perform model evaluation and testing
            test_accuracy = model_eval(model, test_loader, device, batch_size)
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                save_checkpoint(model, epoch, optimizer, best_accuracy)
    return model


def model_eval(model, test_loader, device, batch_size) -> float:
    """
    perform model evaluation and testing using the test dataset
    :param model:
    :param test_loader:
    :param device:
    :param batch_size:
    :return: accuracy of type float
    """
    # put model in evaluation mode and disengage it from tracking the gradients
    model.eval()
    with torch.no_grad():  # torch.inference_mode():
        # initialize variables
        total_correct = 0
        total_sample = 0
        n_class_correct = [0 for i in range(10)]
        n_class_sample = [0 for i in range(10)]
        # loop over the test set
        for images, labels in test_loader:
            # move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # perform a forward pass
            outputs = model(images)

            # calculate the number of correct predictions
            _, prediction = torch.max(outputs.data, 1)
            total_sample = total_sample + labels.size(0)
            total_correct = total_correct + (prediction == labels).sum().item()
            for i in range(batch_size):
                label = labels[i]
                pred = prediction[i]
                if label == pred:
                    n_class_correct[label] = n_class_correct[label] + 1
                n_class_sample[label] = n_class_sample[label] + 1
        total_accuracy = 100.0 * total_correct / total_sample
        print('accuracy of the network on the 10000 test images: {} %'.format(total_accuracy))

        for i in range(10):
            class_accuracy = 100.0 * n_class_correct[i] / n_class_sample[i]
            print('accuracy is {} class: {} %'.format(i, class_accuracy))
    return total_accuracy


if __name__ == "__main__":
    # load hyper parameters
    print("loading hyper parameters")
    learning_rate, epochs, batch_size = hyperparameter()

    # load dataset
    print("loading training set")
    train_loader, test_loader, classes = prepare_dataset(batch_size)
    print(type(train_loader) == DataLoader, type(test_loader) == Dataset, type(classes))

    # configure device
    print("configure device")
    device = configure_device()

    # instantiate the model
    print("instantiate model")
    model = LeNet().to(device)
    print(type(model) == LeNet)

    # define loss and optimizer
    print("optimizing model")
    loss_fn, optimizer = model.loss_optimizer(lr=learning_rate)
    print(type(loss_fn), type(optimizer))
# torch.optim
# torch.nn.modules.loss
    # train the model
    print("running the training function")
    train(model, train_loader, test_loader, epochs, loss_fn, device, batch_size, optimizer)
