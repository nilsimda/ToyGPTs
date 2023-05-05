#! /usr/bin/env python3
class FloatArithmeticCoding:
    """Arithmetic coding that maps text to a float number in [0, 1] with
    the encode() function. Decode() can be used to get the text back. Should
    not be used for long text as the precision of float numbers is limited,
    but is very useful for understanding the concept of arithmetic coding."""

    def __init__(self, model):
        self.model = model

    def encode(self, file):
        """Encode a string of symbols to a float number in [0, 1]"""
        lb, ub = 0, 1
        with open(file, "r") as f:
            while True:
                symbols = f.read(self.model.context_length)
                if not symbols:
                    break
                p1, p2 = self.model.getProbas(symbols)
                interval = ub - lb
                ub = lb + interval * p2
                lb = lb + interval * p1

        res = (ub + lb) / 2
        with open(f"{file}.cmplm", "w") as f:
            f.write(str(res))

    def decode(self, file, length):
        """Decode a float number in [0, 1] back to a string"""
        with open(file, "r") as f:
            num_encoded = float(f.read())
        lb, ub = 0, 1
        res = ""
        for i in range(length):
            interval = ub - lb
            c = self.model.getSymbol((num_encoded - lb) / interval)
            res += c
            p1, p2 = self.model.getProbas(c)
            ub = lb + interval * p2
            lb = lb + interval * p1

        with open(".".join(file.split(".")[0:-1]), "w") as f:
            f.write(res)


class ArithmeticCoding:
    def encode(self, text):
        ub = 0x0
        lb = 0xFFFFFFFF
        for symbol in text:
            p1, p2 = self._getProbas(symbol)
            interval = ub - lb
            ub = lb + interval * p2
            lb = lb + interval * p1

        return (ub + lb) / 2
