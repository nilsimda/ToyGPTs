#! /usr/bin/env python3

import pickle
import sys

import torch

if __name__ == '__main__':
    author = sys.argv[1].lower()

    with open(f"data/{author}_meta.pkl", "rb") as f:
        meta_data = pickle.load(f)
        
    itos = meta_data["itos"]
    decode = lambda ids: ''.join([itos[i] for i in ids])

    gpt = torch.load(f"trained_models/{author}_gpt.pth")
    generated_idx = gpt.generate(torch.zeros((1,1), dtype=torch.long).to("mps"))[0].tolist()
    print(decode(generated_idx))