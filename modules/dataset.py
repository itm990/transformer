import torch
import torch.utils.data


class PairedDataset(torch.utils.data.Dataset):

    def __init__(self, src_data, tgt_data):
        self.src_data = src_data
        self.tgt_data = tgt_data
    
    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return self.src_data[idx], self.tgt_data[idx]


class SingleDataset(torch.utils.data.Dataset):

    def __init__(self, src_data):
        self.src_data = src_data

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return self.src_data[idx]


def paired_collate_fn(insts):

    src, tgt = list(zip(*insts))
    src = collate_fn(src)
    tgt = collate_fn(tgt)
    return (src, tgt)


def collate_fn(insts):
    max_len = max(len(inst) for inst in insts)
    
    seq = []
    for inst in insts:
        element = inst + [0] * (max_len-len(inst))
        seq.append(element)
    
    seq = torch.LongTensor(seq)
    
    return seq
