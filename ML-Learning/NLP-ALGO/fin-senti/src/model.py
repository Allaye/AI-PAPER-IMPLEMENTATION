import transformers
import torch.nn as nn


class BertUncanned(nn.Module):
    def __init__(self):
        super(BertUncanned, self).__init__()
        import configuration
        self.model = transformers.BertModel.from_pretrained(configuration.PRE_TRAINED_BERT_PATH, return_dict=False)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        _, pooled_output = self.model(ids, attention_mask=mask, token_type_ids=token_type_ids)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
