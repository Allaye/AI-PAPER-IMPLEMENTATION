import transformers


MAX_LENGTH = 312
TRAIN_BATCH_SIZE = 3
VALID_BATCH_SIZE = 2
EPOCHS = 10
ACCUMULATION = 2
PRE_TRAINED_BERT_PATH = "../model_configuration/pre_trained_bert"
MODEL_PATH = "model_weights.bin"
TOKENIZER = transformers.BertTokenizer.from_pretrained(PRE_TRAINED_BERT_PATH, do_lower_case=True)
TRAINING_DATASET = "../model_configuration/dataset.csv"
DEVICE = "cuda"
