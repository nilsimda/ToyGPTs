#! /usr/bin/env python3
import numpy as np


class FloatArithmeticCoding:
    """Arithmetic coding that maps text to a float number in [0, 1] with
    the encode() function. Decode() can be used to get the text back. Should
    not be used for long text as the precision of float numbers is limited,
    but is very useful for understanding the concept of arithmetic coding."""

    def __init__(self, prob_dict):
        """Initialize the mapping between symbols and probability ranges"""
        ranges = np.cumsum([0] + list(prob_dict.values()))
        self.mapping = {
            k: (ranges[i], ranges[i + 1])
            for k, i in zip(prob_dict.keys(), range(len(ranges) - 1))
        }
        self.reverse_mapping = {v: k for k, v in self.mapping.items()}

    def _getProbas(self, symbol):
        """Get the probability range of a symbol"""
        return self.mapping[symbol]

    def _getSymbol(self, p):
        """Get the symbol of a probability range"""
        for k in self.reverse_mapping.keys():
            if k[0] <= p < k[1]:
                return self.reverse_mapping[k]

    def encode(self, symbols):
        """Encode a string of symbols to a float number in [0, 1]"""
        lb, ub = 0, 1
        for symbol in symbols:
            p1, p2 = self._getProbas(symbol)
            interval = ub - lb
            ub = lb + interval * p2
            lb = lb + interval * p1

        return (ub + lb) / 2

    def decode(self, num_encoded, length):
        """Decode a float number in [0, 1] back to a string"""
        lb, ub = 0, 1
        res = ""
        for i in range(length):
            interval = ub - lb
            c = self._getSymbol((num_encoded - lb) / interval)
            res += c
            p1, p2 = self._getProbas(c)
            ub = lb + interval * p2
            lb = lb + interval * p1

        return res


class ArithmeticCoding:
    pass


if __name__ == "__main__":
    ac = FloatArithmeticCoding({"H": 0.2, "E": 0.2, "L": 0.4, "O": 0.2})
    input_str = "HELLLLLO"
    encoded_str = ac.encode(input_str)
    print(f"Encoded string: {encoded_str}")
    decoded_str = ac.decode(encoded_str, len(input_str))
    print(f"Decoded string: {decoded_str}")
