#! /usr/bin/env python3
import numpy as np


class ArithmeticCoding:
    def encode(self, symbols, model):
        lb, ub = 0, 1
        for symbol in symbols:
            p1, p2 = model.predict(symbol)
            interval = ub - lb
            ub = lb + interval * p2
            lb = lb + interval * p1
            print(lb, ub)

        print(ub)
        print(lb)

        # binary search
        def _binary_search(low, high):
            m = (low + high) / 2
            if m < lb:
                return "1" + _binary_search(m, high)
            elif m > ub:
                return "0" + _binary_search(low, m)
            return ""

        return _binary_search(0, 1)

    def decode(self, probs, encoded):
        pass


class DummyModel:
    def __init__(self, prob_dict):
        ranges = np.cumsum([0] + list(prob_dict.values()))
        self.mapping = {
            k: (ranges[i], ranges[i + 1])
            for k, i in zip(prob_dict.keys(), range(len(ranges) - 1))
        }
        print(self.mapping)

    def predict(self, c):
        return self.mapping[c]


if __name__ == "__main__":
    ac = ArithmeticCoding()
    model = DummyModel({"H": 0.2, "E": 0.2, "L": 0.4, "O": 0.2})
    encoded_str = ac.encode("HELLO", model)
    print(encoded_str)
