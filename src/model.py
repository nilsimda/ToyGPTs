import string

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class FixedProbabilityModel:
    def __init__(self, context_length=1):
        """Initialize the mapping between symbols and probability ranges"""

        self.context_length = context_length

        num_c = len(string.printable)
        prob_dict = {c: 1 / num_c for c in string.printable}
        ranges = np.cumsum([0] + list(prob_dict.values()))

        self.mapping = {
            k: (ranges[i], ranges[i + 1])
            for k, i in zip(prob_dict.keys(), range(len(ranges) - 1))
        }
        self.reverse_mapping = {v: k for k, v in self.mapping.items()}

    def getProbas(self, symbol):
        """Get the probability range of a symbol"""
        return self.mapping[symbol]

    def getSymbol(self, p):
        """Get the symbol of a probability range"""
        for k in self.reverse_mapping.keys():
            if k[0] <= p < k[1]:
                return self.reverse_mapping[k]


class GPT2:
    def __init__(self, context_length=32):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
        self.context_length = context_length

    def getProbas(self, context_inputs, next_token_id):
        """Get the probability range of a symbol"""
        with torch.no_grad():
            out = self.model(**context_inputs, labels=context_inputs["input_ids"])
            probs = out.logits[0, -1, :].squeeze().softmax(dim=0)
            values, indices = torch.topk(probs, 10)
            print(values)
            print(indices)
            print(self.tokenizer.decode(indices))
            return probs[next_token_id].item()

    def getSymbol(self, p):
        """Get the symbol of a probability range"""
        for k in self.reverse_mapping.keys():
            if k[0] <= p < k[1]:
                return self.reverse_mapping[k]
