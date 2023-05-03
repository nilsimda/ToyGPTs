#! /usr/bin/env python3
import numpy as np


class ArithmeticCoding:
    def __init__(self, model):
        self.model = model

    def encode(self, symbols):
        lb, ub = 0, 1
        for symbol in symbols:
            p1, p2 = self.model.predict(symbol)
            interval = ub - lb
            ub = lb + interval * p2
            lb = lb + interval * p1

        """
        # binary search
        def _binary_search(low, high):
            m = (low + high) / 2
            if m < lb:
                return "1" + _binary_search(m, high)
            elif m > ub:
                return "0" + _binary_search(low, m)
            return ""

        return _binary_search(0, 1)
        """
        return (ub + lb) / 2

    def decode(self, num_encoded, length):
        lb, ub = 0, 1
        res = ""
        for i in range(length):
            interval = ub - lb
            c = model.getSymbol((num_encoded - lb) / interval)
            res += c
            p1, p2 = model.predict(c)
            ub = lb + interval * p2
            lb = lb + interval * p1

        return res


class DummyModel:
    def __init__(self, prob_dict):
        ranges = np.cumsum([0] + list(prob_dict.values()))
        self.mapping = {
            k: (ranges[i], ranges[i + 1])
            for k, i in zip(prob_dict.keys(), range(len(ranges) - 1))
        }
        self.reverse_mapping = {v: k for k, v in self.mapping.items()}

    def predict(self, c):
        return self.mapping[c]

    def getSymbol(self, p):
        for k in self.reverse_mapping.keys():
            if k[0] <= p < k[1]:
                return self.reverse_mapping[k]


if __name__ == "__main__":
    model = DummyModel({"H": 0.2, "E": 0.2, "L": 0.4, "O": 0.2})
    ac = ArithmeticCoding(model)
    encoded_str = ac.encode("HELLO")
    print(f"Encoded string: {encoded_str}")
    decoded_str = ac.decode(encoded_str, 5)
    print(f"Decoded string: {decoded_str}")
