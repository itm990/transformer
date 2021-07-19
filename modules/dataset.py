import torch
import torch.utils.data


class PairedDataset(torch.utils.data.Dataset):

    def __init__(self, bos_idx, eos_idx, src_data, tgt_data):
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.src_data = src_data
        self.tgt_data = tgt_data
    
    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        encoder_input  = self.src_data[idx]
        decoder_input  = [self.bos_idx] + self.tgt_data[idx]
        decoder_output = self.tgt_data[idx] + [self.eos_idx]
        return encoder_input, decoder_input, decoder_output


class SingleDataset(torch.utils.data.Dataset):

    def __init__(self, src_data):
        self.src_data = src_data

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return self.src_data[idx]


def paired_collate_fn(insts):

    enc_in, dec_in, dec_out = list(zip(*insts))
    enc_in  = collate_fn(enc_in)
    dec_in  = collate_fn(dec_in)
    dec_out = collate_fn(dec_out)
    return (enc_in, dec_in, dec_out)


def collate_fn(insts):
    max_len = max(len(inst) for inst in insts)
    
    seq = []
    for inst in insts:
        element = inst + [0] * (max_len-len(inst))
        seq.append(element)
    
    seq = torch.LongTensor(seq)
    
    return seq
