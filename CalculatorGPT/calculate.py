#! /usr/bin/env python3
import pickle
import re

import torch
from prepare_data import decode, encode

if __name__ == "__main__":
    with open(f"data/meta.pkl", "rb") as f:
        meta_data = pickle.load(f)

    stoi = meta_data["stoi"]
    itos = meta_data["itos"]
    max_digits = meta_data["max_digits"]

    gpt = torch.load('trained_models/calculator_gpt.pth').eval()
    gpt.to('cpu')
    while True:
        problem_str = input("Input:")
        problem_str = problem_str.replace(" ", "")
        idx = encode(f"{problem_str}=", stoi, max_digits)
        out_idx = gpt.calculate(idx.unsqueeze(0))
        out_str = decode(out_idx[0], stoi, itos, max_digits)
        pattern = r'(\d+)([+\-*\/])(\d+)=(\d+)'
        matches = re.match(pattern, out_str)
        res = matches.group(4)
        print(f"Output: {res.lstrip('0')}")




