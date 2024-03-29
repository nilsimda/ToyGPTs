import pickle
import sys

import torch

# expects the complete works of some author as a text file, e.g shakespear.txt
author = sys.argv[1]
with open(f"data/{author.lower()}.txt", "r") as f:
    data = f.read()

num_chars = len(data)
print(f"Number of chars in the complete works of {author.capitalize()}: {num_chars:,}")

chars = sorted(list(set(data)))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda cs: [stoi[c] for c in cs]

vocab_size = len(chars)
print(f"Unique chars in dataset: {vocab_size}")

# create train/val split
n_split = int(0.9 * num_chars)
trainset = torch.tensor(encode(data[:n_split]), dtype=torch.long)
valset = torch.tensor(encode(data[n_split:]), dtype=torch.long)

print(f"Trainset size: {len(trainset):,}")
print(f"Valset size: {len(valset):,}")

# save datasets and some metadata to file
torch.save(trainset, f"data/{author}_train.pt")
torch.save(valset, f"data/{author}_val.pt")

meta_data = {"vocab_size": vocab_size, "stoi": stoi, "itos": itos}
with open(f"data/{author}_meta.pkl", "wb") as f:
    pickle.dump(meta_data, f)
