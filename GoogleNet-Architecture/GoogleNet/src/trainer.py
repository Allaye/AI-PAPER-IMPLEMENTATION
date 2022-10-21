import torch
from dataloader import PrepareDataset
from googlenet import GoogleNet
from configuration import configureDevice, hyperParameter
from utils import saveCheckpoint


class TrainerEvaluation:
    def __init__(self, model, trainDataloader, testDataloader, lossFn, optimizer, hyperParameters: dict):
        self.model = model
        self.trainLoader = trainDataloader
        self.testLoader = testDataloader
        self.lossFN = lossFn
        self.optimizer = optimizer
        self.hyperParameters = hyperParameters

    def train(self):
        # enter training mode
        best_accuracy = 0
        n_total_steps = len(self.trainLoader)
        print(f"Model Training Starting...")
        for epoch in range(self.hyperParameters.epochs):
            self.model.train()
            for i, (images, labels) in enumerate(self.trainLoader):
                # move tensors to appropriate device
                images = images.to(self.hyperParameters.get('device'))
                labels = labels.to(self.hyperParameters.get('device'))

                # perform a forward pass
                outputs = self.model(images)

                # calculate the loss
                loss = self.lossFN(outputs, labels)

                # engage model gradients and perform the backward and optimize step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # print training statistics and other information
                if (i + 1) % 100 == 0:
                    print(
                        f'Epoch [{epoch + 1}/{self.hyperParameters.epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')

                # perform model evaluation and testing
                test_accuracy = self.evaluate()
                if test_accuracy > best_accuracy:
                    best_accuracy = test_accuracy
                    saveCheckpoint(self.model, epoch, self.optimizer, best_accuracy)
        return self.model

    def evaluate(self):

        self.model.eval()
        with torch.inference_mode():
            # initialize variables
            total_correct = 0
            total_sample = 0
            n_class_correct = [0 for i in range(10)]
            n_class_sample = [0 for i in range(10)]
            # loop over the test set
            for images, labels in self.testLoader:
                # move tensors to the configured device
                images = images.to(self.hyperParameters.get('device'))
                labels = labels.to(self.hyperParameters.get('device'))

                # perform a forward pass
                outputs = self.model(images)

                # calculate the model correct predictions
                _, prediction = torch.max(outputs.data, 1)
                total_sample = total_sample + labels.size(0)
                total_correct = total_correct + (prediction == labels).sum().item()
                for i in range(self.hyperParameters.get('batch_size')):
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

    def train_evaluate(self):
        pass
