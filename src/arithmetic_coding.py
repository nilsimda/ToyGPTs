#! /usr/bin/env python3
import ctypes


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
    def __init__(self, model):
        self.model = model

    def encode(self, file):
        pending_bits = 0
        lb = ctypes.c_uint(0)
        ub = ctypes.c_uint(0xFFFFFFFF)
        with open(file, "r") as f, open(f"{file}.cmplm", "w") as out:
            while True:
                symbols = f.read(self.model.context_length)
                if not symbols:
                    break
                print(lb, ub)
                p1, p2 = self.model.getProbas(symbols)
                p1_numer, p1_denom = p1.as_integer_ratio()
                p2_numer, p2_denom = p2.as_integer_ratio()
                interval = ub.value - lb.value + 1
                ub.value = lb.value + (interval * p2_numer) // p2_denom
                lb.value = lb.value + (interval * p1_numer) // p1_denom
                print(f"{lb} {ub}")

                def _write_out(bit):
                    out.write(str(int(bit)))
                    nonlocal pending_bits
                    for _ in range(pending_bits):
                        out.write(str(int(not bit)))
                    pending_bits = 0

                while True:
                    if ub.value < 0x80000000:
                        _write_out(False)
                        lb.value <<= 1
                        ub.value <<= 1
                        ub.value |= 1
                    elif lb.value >= 0x80000000:
                        _write_out(True)
                        lb.value <<= 1
                        ub.value <<= 1
                        ub.value |= 1
                    elif lb.value >= 0x40000000 and ub.value < 0xC0000000:
                        pending_bits += 1
                        lb.value <<= 1
                        lb.value &= 0x7FFFFFFF
                        ub.value <<= 1
                        ub.value |= 0x80000001
                    else:
                        break

    def decode():
        pass
