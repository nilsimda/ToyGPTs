import argparse
import os
import pickle
import re

import torch


def encode(prob_str, stoi, max_digits):
    pattern = r'(\d+)([+\-*\/])(\d+)=(\d*)'
    matches = re.match(pattern, prob_str)

    num1 = int(matches.group(1))
    op = matches.group(2)
    num2 = int(matches.group(3))
    res = matches.group(4)
    res = None if res == "" else int(res)

    # encode symbols
    op_enc = torch.tensor(stoi[op], dtype=torch.long).view(1)
    equals_enc = torch.tensor(stoi['='], dtype=torch.long).view(1)

    # encode input numbers (left pad with zeros until num_digits)
    def _encode_num(num, num_digits, reverse=False):
        if reverse:
            out = torch.tensor([int(digit) for digit in reversed(str(num).zfill(num_digits))], dtype=torch.long)
        else:
            out = torch.tensor([int(digit) for digit in str(num).zfill(num_digits)], dtype=torch.long)
        return out

    num1_enc = _encode_num(num1, max_digits)
    num2_enc = _encode_num(num2, max_digits)

    if res is None:
        out = torch.cat([num1_enc, op_enc, num2_enc, equals_enc])
    else:
        res_enc = _encode_num(res, 2*max_digits) if op == '/' else _encode_num(res, 2*max_digits, reverse=True)
        out = torch.cat([num1_enc, op_enc, num2_enc, equals_enc, res_enc])

    return out

def decode(x, stoi, itos, max_digits):
    out = []
    for idx in x:
        if idx < 10: out.append(str(idx.item())) # if its a digit just add it as a str
        elif idx == stoi['<END>']: break # END token means we are done
        else: out.append(itos[idx.item()]) # otherwise encode op

    out =  "".join(out)
    out = out[:-2*max_digits] + out[:(-2*max_digits)-1:-1] # reverse the result
    return out

# we sample two random numbers as input and their sum as the label
def sample_mathproblems(num_problems, allowed_ops): 
    ops_idx = [stoi[op] for op in allowed_ops]
    ops = torch.randint(min(ops_idx), max(ops_idx)+1, (num_problems, ), dtype=torch.long)
    all_nums = torch.randint(0, 10**max_digits, (num_problems, 2), dtype=torch.long)
    
    x = torch.zeros((num_problems, context_length), dtype=torch.long)

    for i, (nums, op) in enumerate(zip(all_nums, ops)):
        num1, num2 = nums[0].item(), nums[1].item()
        op_c = itos[op.item()]
        match op_c:
            case '+':
                res = num1 + num2
            case '-':
                if num2 > num1: num1, num2 = num2, num1 #swap because we dont want negative numbers
                res = num1 - num2
            case '*':
                res = num1 * num2
            case '/':
                if num2 > num1: num1, num2 = num2, num1
                if num2 == 0: num2 = 1
                res = num1 // num2
        x[i] = encode(f"{num1}{op_c}{num2}={res}", stoi, max_digits)

    input_size = 2 * max_digits + 2 # two numbers, one operator, one equals
    masked_loss = -100 * torch.ones((num_problems, input_size-1), dtype=torch.long)
    end_token = stoi['<END>'] * torch.ones((num_problems, 1), dtype=torch.long)
    y = torch.cat([masked_loss, x[:, input_size:], end_token], dim=1)
    return x, y

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_problems', type=int, default=100_000)
    parser.add_argument('--max_digits', type=int, default=6)
    parser.add_argument('--ops', type=str, default='+-*/')


    stoi = {'+': 10, '-':11, '*': 12, '/': 13, '=': 14, '<END>':15}
    itos = {i: op for op, i in stoi.items()}

    args = parser.parse_args()
    num_problems = args.num_problems
    max_digits = args.max_digits
    context_length = 2 * max_digits + 2 * max_digits + 2 # two numbers, one result (can be 2*max when multiplied), one operator, one equals

    val_size = 0.9
    x, y = sample_mathproblems(num_problems, args.ops)
    x_train, y_train = x[:int(num_problems*val_size)], y[:int(num_problems*val_size)]
    x_val, y_val = x[int(num_problems*val_size):], y[int(num_problems*val_size):]

    print(f"Created trainset of size {len(x_train)}")
    print(f"Created valset of size {len(x_val)}")

    # save data
    os.makedirs('data', exist_ok=True)
    torch.save(x_train, 'data/x_train.pt')
    torch.save(y_train, 'data/y_train.pt')
    torch.save(x_val, 'data/x_val.pt')
    torch.save(y_val, 'data/y_val.pt')

    meta_data = {"max_digits": max_digits, "stoi": stoi, "itos": itos}
    with open(f"data/meta.pkl", "wb") as f:
        pickle.dump(meta_data, f)

