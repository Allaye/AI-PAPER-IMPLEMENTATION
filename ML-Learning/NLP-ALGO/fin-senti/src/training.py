import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import configuration
import data_loader
import model_function
from utils import check_gpu
from utils import clean_gpu_memory
from model import BertUncanned
from sklearn import model_selection
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup


def start_training():
    dfx = pd.read_csv(configuration.TRAINING_DATASET).fillna("none")
    dfx.sentiment = dfx.sentiment.apply(lambda x: 1 if x == "positive" else 0)

    df_train, df_valid = model_selection.train_test_split(
        dfx, test_size=0.1, random_state=42, stratify=dfx.sentiment.values
    )

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    train_dataset = data_loader.DataLoader(
        review=df_train.review.values, target=df_train.sentiment.values
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=configuration.TRAIN_BATCH_SIZE, num_workers=4
    )

    valid_dataset = data_loader.DataLoader(
        review=df_valid.review.values, target=df_valid.sentiment.values
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=configuration.VALID_BATCH_SIZE, num_workers=1
    )

    device = torch.device(configuration.DEVICE)
    model = BertUncanned()
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(df_train) / configuration.TRAIN_BATCH_SIZE * configuration.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    best_accuracy = 0
    for epoch in range(configuration.EPOCHS):
        model_function.training_function(train_data_loader, model, optimizer, device, scheduler)
        outputs, targets = model_function.evaluation_function(valid_data_loader, model, device)
        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs)
        print(f"Accuracy Score = {accuracy}")
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), configuration.MODEL_PATH)
            best_accuracy = accuracy


if __name__ == "__main__":
    check_gpu()
    clean_gpu_memory()
    start_training()
