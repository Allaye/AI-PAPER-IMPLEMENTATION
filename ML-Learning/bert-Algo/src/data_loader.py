import torch
import configuration


class DataLoader:
    def __init__(self, review, target) -> None:
        self.target = target
        self.review = review
        self.tokenizer = configuration.TOKENIZER
        self.max_len = configuration.MAX_LENGTH

    def __len__(self):
        return len(self.review)

    def __getitem__(self, item):
        review = str(self.review[item])
        review = " ".join(review.split())
        # tokenized_review = self.tokenizer.encode_plus(review, None, add_special_tokens=True, max_length=self.max_len)
        # input_ids, attention_mask, token_type_ids = tokenized_review["input_ids"], tokenized_review["attention_mask"], tokenized_review["token_type_ids"]
        inputs = self.tokenizer.encode_plus(review, None, add_special_tokens=True, max_length=self.max_len, pad_to_max_length=True, truncation=True)
        ids, mask, token_type_ids = inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(self.target[item], dtype=torch.float),
        }
