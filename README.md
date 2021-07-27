# Transformer

## 説明

Transformerモデル

## 要件

- Python 3.7.3
- PyTorch 1.6.0
- tqdm 4.56.0
- nltk 3.4.3

## 使用方法

- 学習
```
$ train.py \
    --src_vocab_path [source vocabulary] \
    --tgt_vocab_path [target vocabulary] \
    --src_train_path [source train data] \
    --tgt_train_path [target train data] \
    --src_valid_path [source validation data] \
    --tgt_valid_path [target validation data] \
    --sentence_num 20000 \
    --max_length 50 \
    --batch_size 96 \
    --dropout 0.1 \
    --epoch_size 20 \
    --ffn_hidden_size 2048 \
    --hidden_size 256 \
    --init \
    --label_smoothing 0.1 \
    --max_norm 5.0 \
    --name [model name] \
    --parallel_size 8 \
    --seed 42 \
    --sub_layer_num 6 \
    --use_amp \
    --valid_batch_size 50 \
    --weight_decay 1e-6
```

- 翻訳
```
$ eval.py \
    [model name]/best.pt \
    --src_eval_path [source evaluation data] \
    --batch_size 50 \
    --name [output file name]
```