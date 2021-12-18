import torch
from lenet import LeNet
from dataset_loader import prepare_dataset
from configuration import hyperparameter, configure_device



def train(model, train_loader, test_loader, epochs, loss_fn, device, batch_size, optimizer):
    '''
    perform model training loop and hyperparameter tuning
    '''
    best_accuracy = 0
    n_total_steps = len(train_loader)
    print(f"starting training now!")
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            # move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)
            # perform a forward pass
            outputs = model(images)

            # calculate the loss
            loss = loss_fn(outputs, labels)

            # clear the gradients and perform a backward and optimize step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print training statistics and other information
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')
            # perform model evaluation and testing
            test_accuracy = model_eval(model, test_loader, device, batch_size)
            if (test_accuracy > best_accuracy):
               best_accuracy = test_accuracy
               save_checkpoint(model, epoch, optimizer, best_accuracy)
    return model

def model_eval(model, test_loader, device, batch_size):
    '''
    perform model evaluation and testing using the test dataset
    '''
    # disengage the model from tracking the gradients
    with torch.no_grad():
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


def save_checkpoint(model, epoch, optimizer, best_accuracy):
    check_point = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_accuracy": best_accuracy
    }
    torch.save(check_point, "checkpoint.pth")
    return None



if __name__ == "__main__":
    # load hyperparameters
    print("loading hyperparameters")
    learning_rate, epochs, batch_size = hyperparameter()
    
    # load dataset
    print("loading training set")
    train_loader, test_loader, classes = prepare_dataset(batch_size)

    # configure device
    print("configure device")
    device = configure_device()

    # instanciate the model
    print("instaintiat model")
    model = LeNet().to(device)

    # define loss and optimizer
    print("optimizing model") 
    loss_fn, optimizer = model.loss_optimizer(lr=learning_rate)

    # train the model
    print("runing the training function")
    train(model, train_loader, test_loader, epochs, loss_fn, device, batch_size, optimizer)
