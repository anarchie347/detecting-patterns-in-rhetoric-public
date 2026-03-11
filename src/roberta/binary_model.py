from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class RobertaBinaryClassifier(nn.Module):
    """
    Single-task RoBERTa classifier with one 2-class head.
    """

    def __init__(self, base_model_id: str = "roberta-base"):
        super().__init__()
        self.config = AutoConfig.from_pretrained(base_model_id)
        self.encoder = AutoModel.from_pretrained(base_model_id)

        hidden = self.config.hidden_size
        dropout_p = self.config.hidden_dropout_prob
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(hidden, 2)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.last_hidden_state[:, 0, :])
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {
            "loss": loss,
            "logits": logits,
        }
