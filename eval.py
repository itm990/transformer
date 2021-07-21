import argparse
import json
import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import modules.dataset as dataset
import modules.models as models
from modules.pos_enc import positional_encoding
from modules.translate import translate
from modules.preprocess import load_sentences, convert_sent_to_word, convert_word_to_idx


def main():

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--src_eval_path", type=str, default="../corpus/ASPEC-JE/corpus.tok/test.en")
    
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("model_name", type=str)
    parser.add_argument("--name", type=str, default="output")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    
    model_dir = os.path.dirname(args.model_name)
    with open("{}/config.json".format(model_dir)) as f:
        config = json.load(f)
    config = argparse.Namespace(**config)
    print("model:", args.model_name)
    
    src_dict_data = torch.load(config.src_dict_path)
    tgt_dict_data = torch.load(config.tgt_dict_path)
    src2idx = src_dict_data["dict"]["word2index"]
    idx2tgt = tgt_dict_data["dict"]["index2word"]
    PAD = src2idx["[PAD]"]
    BOS = src2idx["[BOS]"]
    EOS = src2idx["[EOS]"]
    src_dict_size = len(src2idx)
    tgt_dict_size = len(idx2tgt)
    
    # load eval data
    src_eval_sent_list = load_sentences(args.src_eval_path)
    src_eval_sent_list = src_eval_sent_list[:100]
    # convert sent to word
    src_eval_word_list = convert_sent_to_word(src_eval_sent_list)
    
    # convert word to idx
    src_eval_idx_list = convert_word_to_idx(word_list=src_eval_word_list, word2index=src2idx)
    
    eval_data = dataset.SingleDataset(src_data=src_eval_idx_list)
    eval_loader = DataLoader(eval_data,
                             batch_size=args.batch_size,
                             collate_fn=dataset.collate_fn,
                             shuffle=False)

    src_max_len = max([len(element) for element in src_eval_idx_list])
    pos_enc = positional_encoding(src_max_len+50, config.hidden_size)
    
    transformer = models.Transformer(
        PAD,
        config.hidden_size, config.ffn_hidden_size,
        src_dict_size, tgt_dict_size,
        config.parallel_size, config.sub_layer_num,
        config.dropout,
        config.init
    ).to(device)
    
    transformer.load_state_dict(torch.load(args.model_name))
    
    sentence_list = translate(PAD, BOS, EOS,
                              src_max_len, idx2tgt, pos_enc,
                              eval_loader, transformer, device)
    sentences = ""
    for sentence in sentence_list:
        sentences += " ".join(sentence) + "\n"
    with open("{}.tok".format(args.name), mode="w") as output_f:
        output_f.write(sentences)


if __name__ == "__main__":
    main()
