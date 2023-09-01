import pickle
import sys

import torch
import torch.nn.functional as F

from model import GPT


def get_batch(split):
    dataset = trainset if split == "train" else valset
    start_idxs = torch.randint(0, len(dataset) - context_length - 1, (batch_size, ))
    x = [dataset[idx : idx + context_length] for idx in start_idxs]
    y = [dataset[idx + 1 : idx + context_length + 1] for idx in start_idxs]
    x = torch.stack(x, dim=0)
    y = torch.stack(y, dim=0)
    return x, y

@torch.no_grad()
def estimate_loss(eval_iters=200):
    gpt.eval()
    loss_d = {}
    for split in ["train", "val"]:
        lossi = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y  = get_batch(split)
            logits = gpt(x)
            lossi[i] = F.cross_entropy(logits, y)
        loss_d[split] = lossi.mean(dim=-1)
    gpt.train()
    return loss_d

def train(trainsteps=10_000):
    for step in range(trainsteps):
        # forward
        x, y = get_batch("train")
        logits = gpt(x)
        loss = F.cross_entropy(logits, y)

        # backward 
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 10_000 == 0:
            loss_d = estimate_loss()
            print(f"Train Loss: {loss_d['train']}, Val Loss {loss_d['val']}")

if __name__ == "__main__":
    # the author to be trained on should be passed
    author = sys.argv[1].lower()

    # load author and meta data, if they dont exist they need to be created with prepared_data.py
    trainset = torch.load(f"data/{author}_train.pt")
    valset = torch.load(f"data/{author}_val.pt")
    print(len(trainset))

    with open(f"data/{author}_meta.pkl", "rb") as f:
        meta_data = pickle.load(f)

    # Some Hyperparameters for the GPT
    batch_size = 2
    n_dec_layers = 2
    context_length = 32
    n_heads = 4
    emb_dim = 128
    vocab_size = meta_data["vocab_size"]
    lr = 1e-5

    # define model and optimizer
    gpt = GPT(n_dec_layers, vocab_size, context_length, n_heads, emb_dim)
    optimizer = torch.optim.AdamW(gpt.parameters(), lr=1e-5)

    train()
