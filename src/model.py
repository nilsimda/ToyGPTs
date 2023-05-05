import string

import numpy as np


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
