import transformers
import torch
from torch import nn
from torch.nn import CrossEntropyLoss


class PointerBert(nn.Module):
    #==========================================================#
    # Model that has an extra dot product layer above
    # Bert embedding. The index of final output tokens of which 
    # score (the dot product with trained weight vector) is the  
    # highest is the index of the target.
    #==========================================================#
    def __init__(self, bertmodel, max_token_len=128):
        super().__init__()
        self.bert = bertmodel
        self.hidden_size = bertmodel.config.hidden_size
        self.num_labels = max_token_len
        self.dropout = nn.Dropout(bertmodel.config.hidden_dropout_prob)
        self.dotproduct = nn.Linear(self.hidden_size, 1, bias=False)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, position_ids=None, head_mask=None, pointer_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        
        last_hidden_state = outputs[0]
        last_hidden_state = self.dropout(last_hidden_state)

        logits = self.dotproduct(last_hidden_state)
        logits = torch.reshape(logits, (-1, self.num_labels))
        if pointer_mask is not None:
            # Tokens other than the special task tokens newly added
            # can be ignored. Therefore we make the scores of such tokens
            # very small.
            logits = logits * pointer_mask - (1-pointer_mask)*torch.finfo(torch.float32).max

        outputs = (logits,) + outputs[2:] 

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

