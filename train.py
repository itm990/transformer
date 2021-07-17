import datetime
import argparse
from logging import getLogger, INFO, FileHandler, Formatter
from tqdm import tqdm
import nltk
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import modules.dataset as dataset
import modules.models as models
from modules.pos_enc import positional_encoding
from modules.translate import translate
from modules.MyNLLLoss import MyNLLLoss


def load_sentences(data_path):
    sent_list = []
    with open(data_path) as f:
        for sent in f:
            sent_list.append(sent.strip())
    return sent_list


def convert_sent_to_word(sent_list):
    return [ sent.strip().split(" ") for sent in sent_list ]

def trim_list(src_lst, tgt_lst, sent_num=5000, max_len=100):
    trimmed_src_lst, trimmed_tgt_lst = [], []
    item_cnt = 0
    for src, tgt in zip(src_lst, tgt_lst):
        if item_cnt >= sent_num:
            break
        if len(src) > max_len or len(tgt) > max_len:
            continue
        trimmed_src_lst.append(src)
        trimmed_tgt_lst.append(tgt)
        item_cnt += 1
    return trimmed_src_lst, trimmed_tgt_lst


def convert_word_to_idx(word_list, word2index):
    return [ [ word2index[word] if word in word2index else word2index["[UNK]"] for word in words ] for words in word_list ]


def train(tgt_EOS, src_PAD, tgt_PAD, max_len, dictionary, pos_enc, args,
          train_loader, valid_loader, valid_word_data, transformer,
          criterion, optimizer, device, model_name):

    train_logger = getLogger(__name__).getChild("train")

    max_score = 0
    step_num = 1
    warmup_steps = 4000

    for epoch in range(args.epoch_size):
        
        transformer.train()

        pbar = tqdm(train_loader, ascii=True)
        total_loss = 0

        for i, batch in enumerate(pbar):
            
            optimizer.zero_grad()

            source, out_tgt = map(lambda x: x.to(device), batch)
            
            batch_size = source.size(0)

            eos = torch.full((batch_size, 1), tgt_EOS,
                             dtype=torch.int64, device=device)
            pad = torch.tensor(tgt_PAD, device=device)
            in_tgt = torch.where(out_tgt==tgt_EOS, pad, out_tgt)
            in_tgt = torch.cat((eos, in_tgt[:, :-1]), dim=1)
            
            src_sent_len, tgt_sent_len = source.size(1), in_tgt.size(1)
            src_pad_mask = source.eq(src_PAD).unsqueeze(1) 
            tgt_pad_mask = in_tgt.eq(tgt_PAD).unsqueeze(1)

            enc_self_attn_mask = src_pad_mask.expand(-1, src_sent_len, -1)
            dec_self_attn_mask = tgt_pad_mask.expand(-1, tgt_sent_len, -1)
            infer_mask = torch.ones((tgt_sent_len, tgt_sent_len),
                                    dtype=torch.uint8,
                                    device=device).triu(diagonal=1)
            infer_mask = infer_mask.unsqueeze(0).expand(batch_size, -1, -1)
            dec_self_attn_mask = torch.gt(dec_self_attn_mask + infer_mask, 0)
            dec_srctgt_mask = src_pad_mask.expand(-1, tgt_sent_len, -1)
            
            enc_pos_enc = pos_enc[:src_sent_len, :].unsqueeze(0)
            enc_pos_enc = enc_pos_enc.expand(batch_size, -1, -1).to(device)
            dec_pos_enc = pos_enc[:tgt_sent_len, :].unsqueeze(0)
            dec_pos_enc = dec_pos_enc.expand(batch_size, -1, -1).to(device)
            
            output = transformer(source, in_tgt, enc_pos_enc, dec_pos_enc,
                                 enc_self_attn_mask, dec_self_attn_mask,
                                 dec_srctgt_mask)
            
            output = output.view(-1, output.size(-1))
            out_tgt = out_tgt.view(-1)
            
            loss = criterion(output, out_tgt)
            
            total_loss += loss
            loss.backward()

            nn.utils.clip_grad_norm_(transformer.parameters(),
                                     max_norm=args.max_norm)

            lrate = args.hidden_size**(-0.5) * min(step_num**(-0.5),
                                                  step_num*warmup_steps**(-1.5))
            for op in optimizer.param_groups:
                op["lr"] = lrate
            optimizer.step()
            step_num += 1

            pbar.set_description("[epoch:%d] loss:%f"
                                 % (epoch+1, total_loss/(i+1)))

        if epoch >= 0:
            sentences = translate(tgt_EOS, src_PAD, tgt_PAD,
                                  max_len, dictionary, pos_enc,
                                  valid_loader, transformer, device)
            bleu_score = nltk.translate.bleu_score.corpus_bleu(valid_word_data, sentences) * 100
            print("BLEU:", bleu_score)
            if bleu_score > max_score:
                max_score = bleu_score
                model = {
                    "model_options" : args,
                    "model_states"  : transformer.state_dict()
                }
                torch.save(model, model_name)
                print("saved.")
            
        train_logger.info("[epoch:%d] loss:%f BLEU:%f"
        % (epoch+1, total_loss/(i+1), bleu_score))


def main():

    datetime_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--src_dict_path", type=str, default="../corpus/ASPEC-JE/dict/aspec_100k.en-ja.en.dict")
    parser.add_argument("--tgt_dict_path", type=str, default="../corpus/ASPEC-JE/dict/aspec_100k.en-ja.ja.dict")
    parser.add_argument("--src_train_path", type=str, default="../corpus/ASPEC-JE/corpus.tok/train-1.en")
    parser.add_argument("--tgt_train_path", type=str, default="../corpus/ASPEC-JE/corpus.tok/train-1.ja")
    parser.add_argument("--src_valid_path", type=str, default="../corpus/ASPEC-JE/corpus.tok/dev.en")
    parser.add_argument("--tgt_valid_path", type=str, default="../corpus/ASPEC-JE/corpus.tok/dev.ja")
    parser.add_argument("--sentence_num", type=int, default=100000)
    parser.add_argument("--max_length", type=int, default=50)
    
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epoch_size", type=int, default=20)
    parser.add_argument("--ffn_hidden_size", type=int, default=2048)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--init", action="store_false")
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--max_norm", type=float, default=5.0)
    parser.add_argument("--name", type=str)
    parser.add_argument("--parallel_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sub_layer_num", type=int, default=6)
    parser.add_argument("--valid_batch_size", type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    logger = getLogger(__name__)
    if args.name is None:
        log_name = "./log/no_name.log"
        model_name = "./model/no_name.pt"
    else:
        log_name = "./log/{}_{}.log".format(args.name, datetime_str)
        model_name = "./model/{}_{}.pt".format(args.name, datetime_str)
        
    fh = FileHandler(log_name)
    fmt = Formatter("[%(levelname)s] %(asctime)s (%(name)s) - %(message)s")
    logger.setLevel(INFO)
    fh.setLevel(INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    
    logger.info(args)
    print(args)
    src_dict_data = torch.load(args.src_dict_path)
    tgt_dict_data = torch.load(args.tgt_dict_path)
    src2idx = src_dict_data["dict"]["word2index"]
    tgt2idx = tgt_dict_data["dict"]["word2index"]
    idx2tgt = tgt_dict_data["dict"]["index2word"]
    PAD = src2idx["[PAD]"]
    BOS = src2idx["[BOS]"]
    EOS = src2idx["[EOS]"]
    src_dict_size = len(src2idx)
    tgt_dict_size = len(tgt2idx)
    
    # load train data
    src_train_sent_list = load_sentences(args.src_train_path)
    tgt_train_sent_list = load_sentences(args.tgt_train_path)
    src_valid_sent_list = load_sentences(args.src_valid_path)
    tgt_valid_sent_list = load_sentences(args.tgt_valid_path)
    
    # convert sent to word
    src_train_word_list = convert_sent_to_word(src_train_sent_list)
    tgt_train_word_list = convert_sent_to_word(tgt_train_sent_list)
    src_valid_word_list = convert_sent_to_word(src_valid_sent_list)
    tgt_valid_word_list = convert_sent_to_word(tgt_valid_sent_list)
    
    # trim word list
    src_train_word_list, tgt_train_word_list = trim_list(
        src_train_word_list, tgt_train_word_list, sent_num=args.sentence_num, max_len=args.max_length
    )
    
    # convert word to idx
    src_train_idx_list = convert_word_to_idx(word_list=src_train_word_list, word2index=src2idx)
    tgt_train_idx_list = convert_word_to_idx(word_list=tgt_train_word_list, word2index=tgt2idx)
    src_valid_idx_list = convert_word_to_idx(word_list=src_valid_word_list, word2index=src2idx)

    train_data = dataset.PairedDataset(src_data=src_train_idx_list,
                                       tgt_data=tgt_train_idx_list)
    train_loader = DataLoader(train_data,
                              batch_size=args.batch_size,
                              collate_fn=dataset.paired_collate_fn,
                              worker_init_fn=random.seed(args.seed),
                              shuffle=True)
        
    valid_data = dataset.SingleDataset(src_data=src_valid_idx_list)
    valid_loader = DataLoader(valid_data,
                              batch_size=args.valid_batch_size,
                              collate_fn=dataset.collate_fn,
                              worker_init_fn=random.seed(args.seed),
                              shuffle=False)
    valid_word_data = [ [words] for words in tgt_valid_word_list ]
    
    pos_enc = positional_encoding(args.max_length+100, args.hidden_size)

    transformer = models.Transformer(
        PAD, PAD,
        args.hidden_size, args.ffn_hidden_size,
        src_dict_size, tgt_dict_size,
        args.parallel_size, args.sub_layer_num,
        args.dropout,
        args.init
    ).to(device)

    optimizer = optim.Adam(transformer.parameters(),
                           betas=(0.9, 0.98),
                           eps=1e-9,
                           weight_decay=args.weight_decay)

    criterion = MyNLLLoss(smooth_weight=args.label_smoothing,
                          ignore_index=PAD)


    train(EOS, PAD, PAD, args.max_length, idx2tgt, pos_enc, args,
          train_loader, valid_loader, valid_word_data, transformer,
          criterion, optimizer, device, model_name)


if __name__ == "__main__":
    main()