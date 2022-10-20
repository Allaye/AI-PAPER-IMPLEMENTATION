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

        # enter training mode
        best_accuracy = 0
        n_total_steps = len(self.trainLoader)
        print(f"Model Training Starting...")
        for epoch in range(hyperParameters.epochs):
            self.model.train()
            for i, (images, labels) in enumerate(self.trainLoader):
                # move tensors to appropriate device
                self.images = images.to(self.hyperParameters.get('device'))
                self.labels = labels.to(self.hyperParameters.get('device'))

                # perform a forward pass
                self.outputs = self.model(self.images)

                # calculate the loss
                self.loss = self.lossFN(self.outputs, self.labels)

                # engage model gradients and perform the backward and optimize step
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()
                





