import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):

    def __init__(self,
                 pad_index,
                 hidden_size, ffn_hidden_size,
                 src_dict_size, tgt_dict_size,
                 parallel_size, sub_layer_num,
                 dropout, init):
        super(Transformer, self).__init__()

        self.encoder = Encoder(pad_index, hidden_size, ffn_hidden_size,
                               src_dict_size, parallel_size, sub_layer_num,
                               dropout, init)
        self.decoder = Decoder(pad_index, hidden_size, ffn_hidden_size,
                               tgt_dict_size, parallel_size, sub_layer_num,
                               dropout, init)
        
    def forward(self,
                source, target, enc_pos_enc, dec_pos_enc,
                enc_self_mask, dec_self_mask, dec_srctgt_mask):

        encoder_output = self.encoder(source, enc_pos_enc, enc_self_mask)
        decoder_output = self.decoder(target, encoder_output, dec_pos_enc,
                                         dec_self_mask, dec_srctgt_mask)

        return decoder_output


class Encoder(nn.Module):

    def __init__(self,
                 pad_index, hidden_size, ffn_hidden_size, dict_size,
                 parallel_size, sub_layer_num, dropout, init):
        super(Encoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(dict_size, hidden_size,
                                      padding_idx=pad_index)
        self.dropout = nn.Dropout(p=dropout)
        self.encoder_sub_layers = nn.ModuleList(
            [EncoderLayer(hidden_size, ffn_hidden_size, parallel_size, dropout)
             for _ in range(sub_layer_num)]
        )
        
        nn.init.constant_(self.embedding.weight[pad_index], 0)
        if init:
            nn.init.normal_(self.embedding.weight, mean=0, std=hidden_size ** -0.5)
        
    def forward(self, input, pos_enc, self_attn_mask):
        
        embedded = self.embedding(input)
        
        encoded = torch.mul(embedded, self.hidden_size**0.5) + pos_enc
        encoded = self.dropout(encoded)

        layer_input = encoded
        for encoder_sub_layer in self.encoder_sub_layers:
            layer_output = encoder_sub_layer(layer_input, self_attn_mask)
            layer_input = layer_output
        output = layer_output

        return output
    

class Decoder(nn.Module):

    def __init__(self,
                 pad_index, hidden_size, ffn_hidden_size, dict_size,
                 parallel_size, sub_layer_num, dropout, init):
        super(Decoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(dict_size, hidden_size,
                                      padding_idx=pad_index)
        self.dropout = nn.Dropout(p=dropout)
        self.decoder_sub_layers = nn.ModuleList(
            [DecoderLayer(hidden_size, ffn_hidden_size, parallel_size, dropout)
             for _ in range(sub_layer_num)]
        )
        self.out = nn.Linear(hidden_size, dict_size, bias=False)
        self.logsoftmax = nn.LogSoftmax(dim=2)
        
        nn.init.constant_(self.embedding.weight[pad_index], 0)
        if init:
            nn.init.normal_(self.embedding.weight, mean=0, std=hidden_size ** -0.5)
            nn.init.xavier_uniform_(self.out.weight)
              
    def forward(self,
                input, encoder_output, pos_enc, self_attn_mask, srctgt_mask):

        embedded = self.embedding(input)

        encoded = torch.mul(embedded, self.hidden_size**0.5) + pos_enc
        encoded = self.dropout(encoded)
        
        layer_input = encoded
        for decoder_sub_layer in self.decoder_sub_layers:
            layer_output = decoder_sub_layer(layer_input, encoder_output,
                                             self_attn_mask, srctgt_mask)
            layer_input = layer_output
        output = layer_output

        output = self.out(output)
        output = self.logsoftmax(output)

        return output


class EncoderLayer(nn.Module):

    def __init__(self, hidden_size, ffn_hidden_size, parallel_size, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(hidden_size,
                                            parallel_size,
                                            dropout)
        self.feed_forward_net = FeedForwardNetwork(hidden_size, ffn_hidden_size,
                                                   hidden_size, dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.layernorm = nn.LayerNorm(hidden_size)

    def forward(self, input, self_attn_mask):

        heads = self.self_attn(input, input, self_attn_mask)
        heads = self.dropout(heads)
        self_attn_out = self.layernorm(input + heads)

        ffn_out = self.feed_forward_net(self_attn_out)
        ffn_out = self.dropout(ffn_out)
        output = self.layernorm(self_attn_out + ffn_out)

        return output


class DecoderLayer(nn.Module):

    def __init__(self, hidden_size, ffn_hidden_size, parallel_size, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(hidden_size,
                                            parallel_size,
                                            dropout)
        self.srctgt_attn = MultiHeadAttention(hidden_size,
                                              parallel_size,
                                              dropout)
        self.feed_forward_net = FeedForwardNetwork(hidden_size, ffn_hidden_size,
                                                   hidden_size, dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.layernorm = nn.LayerNorm(hidden_size)

    def forward(self, input, encoder_output, self_attn_mask, srctgt_mask):
        
        heads = self.self_attn(input, input, self_attn_mask)
        heads = self.dropout(heads)
        self_attn_out = self.layernorm(input + heads)

        heads = self.srctgt_attn(self_attn_out, encoder_output, srctgt_mask)
        heads = self.dropout(heads)
        srctgt_attn_out = self.layernorm(self_attn_out + heads)

        ffn_out = self.feed_forward_net(srctgt_attn_out)
        ffn_out = self.dropout(ffn_out)
        output = self.layernorm(srctgt_attn_out + ffn_out)

        return output


class MultiHeadAttention(nn.Module):

    def __init__(self, hidden_size, parallel_size, dropout):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.parallel_size = parallel_size
        self.query_linear = nn.Linear(hidden_size, hidden_size)
        self.key_linear = nn.Linear(hidden_size, hidden_size)
        self.value_linear = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(p=dropout)
        self.output_linear = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, query, key_value, mask):

        hidden_size = self.hidden_size
        parallel_size = self.parallel_size
        
        query = self.query_linear(query)
        query = torch.chunk(query, self.parallel_size, dim=2)
        query = torch.cat(query, dim=0)

        key = self.key_linear(key_value)
        key = torch.chunk(key, self.parallel_size, dim=2)
        key = torch.cat(key, dim=0)

        value = self.value_linear(key_value)
        value = torch.chunk(value, self.parallel_size, dim=2)
        value = torch.cat(value, dim=0)
        
        key = torch.transpose(key, 1, 2)
        score = torch.bmm(query, key)
        score = torch.div(score, (hidden_size/parallel_size)**0.5)
        masks = mask.repeat(parallel_size, 1, 1)
        score.masked_fill_(masks, -float("inf"))
        
        weight = self.softmax(score)
        weight = self.dropout(weight)
        heads = torch.bmm(weight, value)
        heads = torch.chunk(heads, parallel_size, dim=0)
        heads = torch.cat(heads, dim=2)
        
        heads = self.output_linear(heads)
        
        return heads


class FeedForwardNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, dropout):
        super(FeedForwardNetwork, self).__init__()
        self.ffn1 = nn.Linear(input_size, hidden_size)
        self.ffn2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, input):
        output = self.ffn1(input)
        output = F.relu(output)
        output = self.dropout(output)
        output = self.ffn2(output)
        return output
