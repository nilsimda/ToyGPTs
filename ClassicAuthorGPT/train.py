import pickle
import sys

import torch
import torch.nn.functional as F

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
            losses[i] = loss
        loss_d[split] = losses.mean()
    gpt.train()
    return loss_d

def train(trainsteps=40_000):
    for step in range(trainsteps):
        # forward
        x, y = get_batch("train")
        x, y = x.to(device), y.to(device)
        logits, loss = gpt(x, y)

        # backward 
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 10_000 == 0:
            loss_d = estimate_loss()
            print(f"Train Loss: {loss_d['train']:.4f}, Val Loss {loss_d['val']:.4f}")

if __name__ == "__main__":
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(device)
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

    # Some Hyperparameters for the GPT
    batch_size = 32
    n_dec_layers = 2
    context_length = 32
    n_heads = 4
    emb_dim = 128
    lr = 1e-5

    # define model and optimizer, move model to correct device
    gpt = GPT(n_dec_layers, vocab_size, context_length, n_heads, emb_dim).to(device)
    optimizer = torch.optim.AdamW(gpt.parameters(), lr=1e-5)

    train(30_000)
    generated_idx = gpt.generate(torch.zeros((1,1), dtype=torch.long))[0].tolist()
    print(decode(generated_idx))
