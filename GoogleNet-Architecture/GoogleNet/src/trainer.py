import torch
from dataloader import CustomDataLoader
from torchvision.transforms import transforms
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
        for epoch in range(self.hyperParameters.get('epochs')):
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
                        f'Epoch [{epoch + 1}/{self.hyperParameters.get("epochs")}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')

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
            validation_loss = 0.0
            total_sample = 0
            total_correct = 0
            # loop over the test set
            for images, labels in self.testLoader:
                # move tensors to the configured device
                images = images.to(self.hyperParameters.get('device'))
                labels = labels.to(self.hyperParameters.get('device'))

                # perform a forward pass
                outputs = self.model(images)

                # calculate the model prediction and metrics
                validation_loss += self.lossFN(outputs, labels).item()
                _, prediction = torch.max(outputs.data, 1)
                total_sample = total_sample + labels.size(0)
                total_correct = total_correct + (prediction == labels).sum().item()

            average_validation_loss = validation_loss / len(self.testLoader)
            print(f'Validation Info: Avg_validation_loss: {average_validation_loss:.4f} Validation_loss: {validation_loss:.4f}')
            return 100 * total_correct / total_sample

    def train_evaluate(self):
        pass


if __name__ == "__main__":
    hyperParameter = hyperParameter()
    config = configureDevice()
    hyperParameter.update(config)
    d_path = "./data/dataset"
    data = CustomDataLoader(d_path, transform=[transforms.ToTensor(), transforms.Resize((224, 224))])
    trainDataloader, testDataloader  = data.getdataloader()
    print(hyperParameter)
    loss, optimizer = GoogleNet().loss_optimizer()
    te = TrainerEvaluation(model=GoogleNet(), trainDataloader=trainDataloader, testDataloader=testDataloader, lossFn=loss, optimizer=optimizer, hyperParameters=hyperParameter)
    te.train()