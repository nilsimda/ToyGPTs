#! /usr/bin/env python3
import torch
import yaml
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
            _, loss = model(X.to(device), Y.to(device))
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train_gpt(model, optimizer, train_steps=100_000, eval_iters=200):
    for step in range(train_steps):
        # forward pass
        idx = torch.randint(0, len(x_train), (batch_size,))
        x, y = x_train[idx], y_train[idx]
        _, loss = model(x.to(device), y.to(device))

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
    
    with open('config.yml' , 'r') as f:
        params = yaml.safe_load(f)

    device = torch.device(params['device'])
    batch_size = params['batch_size']
    n_dec_layers = params['n_dec_layers']
    vocab_size = 16 # 10 digits, 4 operators, 1 equals, 1 end token
    n_heads = params['n_heads']
    emb_dim = params['emb_dim']
    context_length = 2 * 6 + 2 * 6 + 2 # two numbers, one result (can be 2*max when multiplied), one operator, one equals
    lr = params['lr']

    x_train = torch.load('data/x_train.pt')
    y_train = torch.load('data/y_train.pt')

    x_val = torch.load('data/x_val.pt')
    y_val = torch.load('data/y_val.pt')
    
    gpt = GPT(n_dec_layers, vocab_size, context_length, n_heads, emb_dim)
    gpt.to(device)
    optimizer = torch.optim.AdamW(gpt.parameters(), lr=lr)
    train_gpt(gpt, optimizer, train_steps=params['trainsteps'])

    print("Training finished! Saving model...")
    torch.save(gpt, f"trained_models/calculator_plus_minus_gpt.pth")
    print("Done!")
