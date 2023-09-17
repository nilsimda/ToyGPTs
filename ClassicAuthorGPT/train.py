import pickle
import sys

import torch
import torch.nn.functional as F
import yaml
from model import GPT


def get_batch(split):
    dataset = trainset if split == "train" else valset
    start_idxs = torch.randint(0, len(dataset) - context_length, (batch_size, ))
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
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y  = get_batch(split)
            x, y = x.to(device), y.to(device)
            _, loss = gpt(x, y)
            losses[i] = loss.item()
        loss_d[split] = losses.mean()
    gpt.train()
    return loss_d

def train(trainsteps=40_000):
    for step in range(trainsteps):
        # forward
        x, y = get_batch("train")
        x, y = x.to(device), y.to(device)
        _, loss = gpt(x, y)

        # backward 
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 10_000 == 0:
            loss_d = estimate_loss()
            print(f"{step}/{trainsteps} Train Loss: {loss_d['train']:.4f}, Val Loss {loss_d['val']:.4f}")

if __name__ == "__main__":
    # the author to be trained on should be passed
    author = sys.argv[1].lower()

    # load author and meta data, if they dont exist they need to be created with prepared_data.py
    trainset = torch.load(f"data/{author}_train.pt")
    valset = torch.load(f"data/{author}_val.pt")

    with open(f"data/{author}_meta.pkl", "rb") as f:
        meta_data = pickle.load(f)

    vocab_size = meta_data["vocab_size"]
    itos = meta_data["itos"]
    decode = lambda ids: ''.join([itos[i] for i in ids])

    with open('config.yml' , 'r') as f:
        params = yaml.safe_load(f)

    # set device from config
    device = torch.device(params["device"]) 
    print(f"Using device: {device}")

    # Set Hyperparameters for the GPT from config
    batch_size = params["batch_size"]
    n_dec_layers = params["n_dec_layers"]
    context_length = params["context_length"]
    n_heads = params["n_heads"]
    emb_dim = params["emb_dim"]
    lr = params["lr"]


    # define model and optimizer, move model to correct device
    gpt = GPT(n_dec_layers, vocab_size, context_length, n_heads, emb_dim).to(device)
    optimizer = torch.optim.AdamW(gpt.parameters(), lr=1e-5)

    train(30_000)
    torch.save(gpt, f"trained_models/{author}_gpt.pth")
