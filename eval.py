import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import modules.dataset as dataset
import modules.models as models
from modules.pos_enc import positional_encoding
from modules.translate import translate


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cuda_num', type=int, default=0)
    parser.add_argument('-e', '--eval_batch_size', type=int, default=50)
    parser.add_argument('-n', '--name', type=str, default='output')
    parser.add_argument('model_name', type=str)
    opt = parser.parse_args()

    device = torch.device('cuda', opt.cuda_num)
    print('device:', device)
    
    model = torch.load(opt.model_name)
    options = model['model_options']
    print(options)
    data = torch.load(options.data_path)
    src_PAD = data['word2index']['src']['<PAD>']
    tgt_PAD = data['word2index']['tgt']['<PAD>']
    tgt_EOS = data['word2index']['tgt']['<EOS>']
    idx2jpn = data['index2word']['tgt']
    src_dict_size = len(data['word2index']['src'])
    tgt_dict_size = len(data['word2index']['tgt'])

    src_max_len = max([len(element) for element in data['eval']['src']])
    pos_enc = positional_encoding(src_max_len+50, options.hidden_size)
    
    transformer = models.Transformer(
        src_PAD, tgt_PAD,
        options.hidden_size, options.ffn_hidden_size,
        src_dict_size, tgt_dict_size,
        options.parallel_size, options.sub_layer_num,
        options.dropout,
        options.init
    ).to(device)
        
    states  = model['model_states']
    transformer.load_state_dict(states)
    
    eval_data = dataset.SingleDataset(src_data=data['eval']['src'])
    eval_loader = DataLoader(eval_data,
                             batch_size=opt.eval_batch_size,
                             collate_fn=dataset.collate_fn,
                             shuffle=False)

    sentence_list = translate(tgt_EOS, src_PAD, tgt_PAD,
                              src_max_len, idx2jpn, pos_enc,
                              eval_loader, transformer, device)
    sentences = ''
    for sentence in sentence_list:
        sentences += ' '.join(sentence) + '\n'
    with open('../sent/{}.txt'.format(opt.name), mode='w') as output_f:
        output_f.write(sentences)


if __name__ == '__main__':
    main()
