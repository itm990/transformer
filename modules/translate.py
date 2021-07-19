from tqdm import tqdm
import torch
import torch.nn as nn


def translate(PAD, BOS, EOS, max_len, dictionary, pos_enc,
              data_loader, transformer, device):

    transformer.eval()

    pbar = tqdm(data_loader, ascii=True)
    sentence_list = []

    with torch.no_grad():
        for batch in pbar:
            
            source = batch.to(device)
            
            batch_size = source.size(0)
            
            in_tgt = torch.full((batch_size, 1), BOS,
                             dtype=torch.int64, device=device)
            eos = torch.full((batch_size, 1), EOS,
                             dtype=torch.int64, device=device)
            
            src_sent_len = source.size(1)
            src_pad_mask = source.eq(PAD).unsqueeze(1)
            
            enc_self_attn_mask = src_pad_mask.expand(-1, src_sent_len, -1)
            
            enc_pos_enc = pos_enc[:src_sent_len, :].unsqueeze(0)
            enc_pos_enc = enc_pos_enc.expand(batch_size, -1, -1).to(device)
            
            encoder_output = transformer.encoder(source,
                                                 enc_pos_enc,
                                                 enc_self_attn_mask)
            
            generated_words = [[] for _ in range(batch_size)]
            original_indices = torch.arange(batch_size, device=device)
            
            for _ in range(max_len + 50):
                
                sent_num = in_tgt.size(0)
                tgt_sent_len = in_tgt.size(1)
                tgt_pad_mask = in_tgt.eq(PAD).unsqueeze(1)
                
                dec_self_attn_mask = tgt_pad_mask.expand(-1, tgt_sent_len, -1)
                infer_mask = torch.ones((tgt_sent_len, tgt_sent_len),
                                        dtype=torch.uint8,
                                        device=device).triu(diagonal=1)
                infer_mask = infer_mask.unsqueeze(0).expand(sent_num, -1, -1)
                dec_self_attn_mask = torch.gt(dec_self_attn_mask + infer_mask, 0)
                dec_srctgt_mask = src_pad_mask.expand(-1, tgt_sent_len, -1)
                
                dec_pos_enc = pos_enc[:tgt_sent_len, :].unsqueeze(0)
                dec_pos_enc = dec_pos_enc.expand(sent_num, -1, -1).to(device)
                
                output = transformer.decoder(in_tgt,
                                             encoder_output,
                                             dec_pos_enc,
                                             dec_self_attn_mask,
                                             dec_srctgt_mask)
            
                predicted_words = output[:, -1:, :]
                predicted_words = torch.argmax(predicted_words, dim=2)
                
                is_eos = predicted_words.eq(eos).squeeze(1)
                is_eos_indices_list = is_eos.nonzero().squeeze(1).tolist()
                not_eos = predicted_words.ne(eos).squeeze(1)
                not_eos_indices = not_eos.nonzero().squeeze(1)
                
                to_original_indices = original_indices.tolist()
                for i in is_eos_indices_list:
                    dict_indices = in_tgt[i, 1:].tolist()
                    words = [dictionary[dict_index] for dict_index in dict_indices]
                    sent_index = to_original_indices[i]
                    generated_words[sent_index] = words
                    
                src_pad_mask = src_pad_mask.index_select(dim=0, index=not_eos_indices)
                encoder_output = encoder_output.index_select(dim=0, index=not_eos_indices)
                in_tgt = in_tgt.index_select(dim=0, index=not_eos_indices)
                original_indices = original_indices.index_select(dim=0, index=not_eos_indices)
                eos = eos.index_select(dim=0, index=not_eos_indices)
                predicted_words = predicted_words.index_select(dim=0, index=not_eos_indices)
                
                # input がなくなれば終了
                if in_tgt.size(0) == 0:
                    break
            
                in_tgt = torch.cat((in_tgt, predicted_words), dim=1)
                
            # 残りの文生成
            sent_num = in_tgt.size(0)
            if sent_num != 0:
                to_original_indices = original_indices.tolist()
                for i in range(sent_num):
                    dict_indices = in_tgt[i, 1:].tolist()
                    words = [dictionary[dict_index] for dict_index in dict_indices]
                    sent_index = to_original_indices[i]
                    generated_words[sent_index] = words
                    
            for sentence in generated_words:
                sentence_list.append(sentence)
        
            pbar.set_description('[translation]')
        
    return sentence_list

    
