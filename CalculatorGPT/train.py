#! /usr/bin/env python3
import torch
from model import GPT


@torch.no_grad()
def estimate_loss(model, eval_iters):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            if split == 'train':
                idx = torch.randint(0, len(x_train), (batch_size, ))
                X, Y = x_train[idx], y_train[idx]
            elif split == 'val':
                idx = torch.randint(0, len(x_val), (batch_size, ))
                X, Y = x_val[idx], y_val[idx]
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train_gpt(model, optimizer, train_steps=100_000, eval_iters=200):
    for step in range(train_steps):
        # forward pass
        idx = torch.randint(0, len(x_train), (batch_size,))
        x, y = x_train[idx], y_train[idx]
        _, loss = model(x, y)

        # backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 10_000 == 0:
            losses = estimate_loss(model, eval_iters) 
            train_loss = losses['train']
            val_loss = losses['val']
            print(f"{step}/{train_steps} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

if __name__ == "__main__":
    batch_size = 32
    n_blocks = 2
    vocab_size = 16
    n_heads = 4
    emb_dim = 128
    context_length = 2 * 6 + 2 * 6 + 2 # two numbers, one result (can be 2*max when multiplied), one operator, one equals

    x_train = torch.load('data/x_train.pt')
    y_train = torch.load('data/y_train.pt')

    x_val = torch.load('data/x_val.pt')
    y_val = torch.load('data/y_val.pt')
    
    gpt = GPT(n_blocks, vocab_size, context_length, n_heads, emb_dim)
    train_gpt(gpt, torch.optim.AdamW(gpt.parameters(), lr=1e-5))

    print("Training finished! Saving model...")
    torch.save(gpt, f"trained_models/calculator_plus_minus_gpt.pth")
    print("Done!")
