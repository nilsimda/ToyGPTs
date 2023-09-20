#! /usr/bin/env python3
import torch
from prepare_data import decode, encode

if __name__ == "__main__":
    problem_str = input("Input:")
    idx = encode(problem_str)
    gpt = torch.load('model.pt').eval()
    out_idx = gpt.calculate(idx.unsqueeze(0))
    print(f"Output: {decode(out_idx[0])}")



