import torch


def encode(num1, num2, op, res=None):
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

    if res == None:
        out = torch.cat([num1_enc, op_enc, num2_enc, equals_enc])
    else:
        res_enc = _encode_num(res, 2*max_digits, True)
        out = torch.cat([num1_enc, op_enc, num2_enc, equals_enc, res_enc])

    return out

def decode(x):
    out = []
    for idx in x:
        if idx < 10: out.append(str(idx.item())) # if its a digit just add it as a str
        elif idx == stoi['<END>']: break # END token means we are done
        else: out.append(itos[idx.item()]) # otherwise encode op

    out =  "".join(out)
    out = out[:-2*max_digits] + out[:(-2*max_digits)-1:-1] # reverse the result
    return out

# we sample two random numbers as input and their sum as the label
def sample_mathproblems(num_problems): 
    ops = torch.randint(10, 12, (num_problems, ), dtype=torch.long)
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
        x[i] = encode(num1, num2, op_c, res)

    input_size = 2 * max_digits + 2 # two numbers, one operator, one equals
    masked_loss = -100 * torch.ones((num_problems, input_size-1), dtype=torch.long)
    end_token = stoi['<END>'] * torch.ones((num_problems, 1), dtype=torch.long)
    y = torch.cat([masked_loss, x[:, input_size:], end_token], dim=1)
    return x, y

if __name__ == "__main__":
    max_digits = 6
    num_problems = 100_000
    context_length = 2 * max_digits + 2 * max_digits + 2 # two numbers, one result (can be 2*max when multiplied), one operator, one equals
    stoi = {'+': 10, '-':11, '=': 14, '<END>':15}
    itos = {i: op for op, i in stoi.items()}

    val_size = 0.9
    x, y = sample_mathproblems(num_problems)
    x_train, y_train = x[:int(num_problems*val_size)], y[:int(num_problems*val_size)]
    x_val, y_val = x[int(num_problems*val_size):], y[int(num_problems*val_size):]

    # save data
    torch.save(x_train, 'data/x_train.pt')
    torch.save(y_train, 'data/y_train.pt')
    torch.save(x_val, 'data/x_val.pt')
    torch.save(y_val, 'data/y_val.pt')

