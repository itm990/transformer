
def make_dict(vocab_path):
    word2index, index2word = {}, {}
    with open(vocab_path) as f:
        for index, word in enumerate(f):
            word = word.strip()
            word2index[word] = index
            index2word[index] = word
    return word2index, index2word


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
