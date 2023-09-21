#! /usr/bin/env python3
import re

import torch
from prepare_data import decode, encode

if __name__ == "__main__":
    gpt = torch.load('trained_models/calculator_gpt.pth').eval()
    gpt.to('cpu')
    while True:
        problem_str = input("Input:")
        problem_str = problem_str.replace(" ", "")
        idx = encode(f"{problem_str}=")
        out_idx = gpt.calculate(idx.unsqueeze(0))
        out_str = decode(out_idx[0])
        pattern = r'(\d+)([+\-*\/])(\d+)=(\d+)'
        matches = re.match(pattern, out_str)
        res = matches.group(4)
        print(f"Output: {res.lstrip('0')}")




