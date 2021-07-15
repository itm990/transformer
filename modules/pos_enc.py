import torch


def positional_encoding(sentence_length, hidden_size):
    dim_num = torch.arange(hidden_size,dtype=torch.float)
    dim_num = dim_num.unsqueeze(dim=0).expand(sentence_length, -1)
    pos_num = torch.arange(sentence_length, dtype=torch.float)
    pos_num = pos_num.unsqueeze(dim=1).expand(-1, hidden_size)
    denom = torch.where(dim_num%2==0,
                        torch.pow(10000, dim_num/hidden_size),
                        torch.pow(10000, (dim_num-1)/hidden_size))
    pe = torch.where(dim_num%2==0,
                     torch.sin(pos_num/denom),
                     torch.cos(pos_num/denom))
    
    return pe
