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


def train(tgt_EOS, src_PAD, tgt_PAD, max_len, dictionary, pos_enc, opt,
          train_loader, valid_loader, valid_word_data, transformer,
          criterion, optimizer, device, model_name):

    train_logger = getLogger(__name__).getChild('train')

    max_score = 0
    step_num = 1
    warmup_steps = 4000

    for epoch in range(opt.epoch_size):
        
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
                                     max_norm=opt.max_norm)

            lrate = opt.hidden_size**(-0.5) * min(step_num**(-0.5),
                                                  step_num*warmup_steps**(-1.5))
            for op in optimizer.param_groups:
                op['lr'] = lrate
            optimizer.step()
            step_num += 1

            pbar.set_description('[epoch:%d] loss:%f'
                                 % (epoch+1, total_loss/(i+1)))

        if epoch >= 0:
            sentences = translate(tgt_EOS, src_PAD, tgt_PAD,
                                  max_len, dictionary, pos_enc,
                                  valid_loader, transformer, device)
            bleu_score = nltk.translate.bleu_score.corpus_bleu(valid_word_data, sentences) * 100
            print('BLEU:', bleu_score)
            if bleu_score > max_score:
                max_score = bleu_score
                model = {
                    'model_options' : opt,
                    'model_states'  : transformer.state_dict()
                }
                torch.save(model, model_name)
                print('saved.')
            
        train_logger.info('[epoch:%d] loss:%f BLEU:%f'
        % (epoch+1, total_loss/(i+1), bleu_score))


def main():

    datetime_str = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=96)
    parser.add_argument('-c', '--cuda_num', type=int, default=0)
    parser.add_argument('--data_path', type=str, default='../data/100000.dict')
    parser.add_argument('-d', '--dropout', type=float, default=0.1)
    parser.add_argument('-e', '--epoch_size', type=int, default=20)
    parser.add_argument('-f', '--ffn_hidden_size', type=int, default=2048)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('-i','--init', action='store_false')
    parser.add_argument('-l', '--label_smoothing', type=float, default=0.1)
    parser.add_argument('-m', '--max_norm', type=float, default=5.0)
    parser.add_argument('-n', '--name', type=str)
    parser.add_argument('-p', '--parallel_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('-s', '--sub_layer_num', type=int, default=6)
    parser.add_argument('-v', '--valid_batch_size', type=int, default=50)
    parser.add_argument('-w', '--weight_decay', type=float, default=1e-6)
    opt = parser.parse_args()
    
    device = torch.device('cuda', opt.cuda_num)
    print('device:', device)

    logger = getLogger(__name__)
    if opt.name is None:
        log_name = './log/no_name.log'
        model_name = './model/no_name.model'
    else:
        log_name = './log/{}_{}.log'.format(opt.name, datetime_str)
        model_name = './model/{}_{}.model'.format(opt.name, datetime_str)
        
    fh = FileHandler(log_name)
    fmt = Formatter('[%(levelname)s] %(asctime)s (%(name)s) - %(message)s')
    logger.setLevel(INFO)
    fh.setLevel(INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    
    logger.info(opt)
    print(opt)
    data = torch.load(opt.data_path)
    src_PAD = data['word2index']['src']['<PAD>']
    tgt_PAD = data['word2index']['tgt']['<PAD>']
    tgt_EOS = data['word2index']['tgt']['<EOS>']
    idx2jpn = data['index2word']['tgt']
    src_dict_size = len(data['word2index']['src'])
    tgt_dict_size = len(data['word2index']['tgt'])
    
    src_max_len = max([len(element) for element in data['valid']['src']])
    pos_enc = positional_encoding(src_max_len+50, opt.hidden_size)

    # seed 固定
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
        
    transformer = models.Transformer(
        src_PAD, tgt_PAD,
        opt.hidden_size, opt.ffn_hidden_size,
        src_dict_size, tgt_dict_size,
        opt.parallel_size, opt.sub_layer_num,
        opt.dropout,
        opt.init
    ).to(device)

    optimizer = optim.Adam(transformer.parameters(),
                           betas=(0.9, 0.98),
                           eps=1e-9,
                           weight_decay=opt.weight_decay)

    criterion = MyNLLLoss(smooth_weight=opt.label_smoothing,
                          ignore_index=tgt_PAD)

    train_data = dataset.PairedDataset(src_data=data['train']['src'],
                                       tgt_data=data['train']['tgt'])
    train_loader = DataLoader(train_data,
                              batch_size=opt.batch_size,
                              collate_fn=dataset.paired_collate_fn,
                              worker_init_fn=random.seed(opt.seed),
                              shuffle=True)
        
    valid_data = dataset.SingleDataset(src_data=data['valid']['src'])
    valid_loader = DataLoader(valid_data,
                              batch_size=opt.valid_batch_size,
                              collate_fn=dataset.collate_fn,
                              worker_init_fn=random.seed(opt.seed),
                              shuffle=False)
    valid_tgt_word_data = data['valid']['tgt_word']

    train(tgt_EOS, src_PAD, tgt_PAD, src_max_len, idx2jpn, pos_enc, opt,
          train_loader, valid_loader, valid_tgt_word_data, transformer,
          criterion, optimizer, device, model_name)


if __name__ == '__main__':
    main()
