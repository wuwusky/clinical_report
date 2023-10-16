# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 20:05:24 2023

@author: Admin
"""
import torch.nn as nn
import torch
import torch.nn.functional as F


import numpy as np
# from collections import namedtuple
# Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TranslationModel(nn.Module):
    def __init__(self, input_l, output_l, n_token, 
                 encoder_layer=6, decoder_layer=6, d=512, n_head=8, dim_ff=2048, 
                 sos_id=1, pad_id=0, 
                 dropout_en=0.1, dropout_de=0.1, dropout=0.0):
        super().__init__()
        self.encoder = Encoder(input_l, n_token, n_layer=encoder_layer, d=d, n_head=n_head, pad_id=pad_id, dim_ff=dim_ff, dropout_en=dropout_en, dropout=dropout)
        self.decoder = Decoder(output_l, input_l, n_token, n_layer=decoder_layer, d=d, n_head=n_head, sos_id=sos_id, pad_id=pad_id, dim_ff=dim_ff, dropout_de=dropout_de, dropout=dropout)
        self.output_pre = nn.Linear(d, n_token)
        self.output_pre_del = nn.Linear(d, n_token)
    
    def forward(self, inputs, outputs=None, beam=1, mode='greedy'):
        feature = self.encoder(inputs) #[B,S,512]
        # if self.training:
        #     feature = F.dropout(feature, p=0.25)
        if outputs is None:
            return self.decoder(feature, caption=None, top_k=beam, mode=mode)
        # return self.decoder(feature, outputs) #[B,L,n_token]
        return self.decoder(feature, outputs)
    
    def pre_en_forward(self, inputs):
        feature = self.encoder(inputs)
        pre_out = self.output_pre(feature)
        return pre_out

    def del_en_forward(self, inputs):
        feature = self.encoder(inputs)
        pre_out = self.output_pre_del(feature)
        return pre_out

    
    def pre_de_forward(self, inputs):
        pass



class Encoder(nn.Module):
    def __init__(self, max_l, n_token, n_layer=6, d=512, n_head=8, pad_id=0, dim_ff=2048, dropout_en=0.1, dropout=0.0):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d, nhead=n_head, activation=F.gelu, dim_feedforward=dim_ff, dropout=dropout_en)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layer) 
        self.posit_embedding = nn.Embedding(max_l, d)
        self.token_embedding = nn.Embedding(n_token, d)
        self.nhead = n_head
        self.pad_id = pad_id

    def forward(self, inputs):
        posit_index = torch.arange(inputs.shape[1]).unsqueeze(0).repeat(inputs.shape[0], 1).to(inputs.device) #(B,S)
        source_posit_embed = self.posit_embedding(posit_index) 
        padding_mask = (inputs == self.pad_id) 
        
        inputs = self.token_embedding(inputs) #[B,S,512]
        
        source_embed = inputs + source_posit_embed
        source_embed = torch.transpose(source_embed, 0, 1)
        attn_mask = torch.full((inputs.shape[1], inputs.shape[1]),0.0).to(inputs.device)

        output = self.transformer_encoder(src=source_embed, mask=attn_mask, src_key_padding_mask=padding_mask) #[S, B, 512]
        output = torch.transpose(output, -2, -3) #[B, S, 512]
        return output
    
class Decoder(nn.Module):
    def __init__(self, max_l, input_l, n_token, sos_id=1, pad_id=0, n_layer=6, n_head=8, d=512, dim_ff=2048, dropout_de=0.1, dropout=0.0):
        super().__init__()
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.n_head = n_head
        self.d = d
        if n_token is not None:
            self.n_token = n_token
            self.token_embedding = nn.Embedding(n_token, d)
        self.posit_embedding = nn.Embedding(max_l, d)
        self.source_posit_embedding = nn.Embedding(input_l, d)
        
        self.max_l = max_l
    
        decoder_layer = nn.TransformerDecoderLayer(d_model=d, nhead=n_head, activation=F.gelu, dim_feedforward=dim_ff, dropout=dropout_de)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layer) 
        
        if n_token is not None:
            self.output = nn.Linear(d, n_token)
            self.output.weight = self.token_embedding.weight

            
    def forward(self, source, caption, top_k=1, eos_id=2, mode='greedy'):
        """
        source: [B,S,E], S=1 or n_slice
        caption: [B,L], token index。
        """
        if caption is None:
            if mode == 'greedy':
                return self._infer(source, top_k=top_k, eos_id=eos_id, mode=mode)
            elif mode == 'beam':
                return self._infer_beam_batch(source, top_k=top_k, eos_id=eos_id, mode=mode) # (B,l)
            else:
                return self._infer(source, top_k=top_k, eos_id=eos_id, mode=mode)
            
        posit_index = torch.arange(caption.shape[1]).unsqueeze(0).repeat(caption.shape[0],1).to(caption.device) #(B,L)
        target_embed = self.posit_embedding(posit_index) #输入shape后面增加E。(B,L,E)
        target_embed += self.token_embedding(caption) # (B,L,E)
        padding_mask = (caption == self.pad_id) #[B,L]
        
        attn_mask = self.generate_square_subsequent_mask(caption.shape[1]).to(caption.device) #[L,L]

        #posit_index = torch.arange(source.shape[1]).unsqueeze(0).repeat(caption.shape[0],1).to(source.device) #(B,S)
        #source_posit_embed = self.source_posit_embedding(posit_index) # [B,S,E]
        #source_embed = source + source_posit_embed
        
        target_embed = torch.transpose(target_embed, 0, 1) 
        source_embed = torch.transpose(source, 0, 1)
        out = self.transformer_decoder(tgt=target_embed, memory=source_embed, tgt_mask=attn_mask, tgt_key_padding_mask=padding_mask)
        out = torch.transpose(out, -2, -3) #[B, L, E]
        out = self.output(out) #[B, L, n_token]
        return out

    def _infer(self, source, top_k=1, eos_id=2, mode='greedy'):
        """
        source: [B,S,E],
        """
        outputs = torch.ones((source.shape[0], 1), dtype=torch.long).to(source.device) * self.sos_id # (K,B,1) SOS
        not_over = torch.ones((source.shape[0])).to(source.device) #[K,B]
        assert top_k==1
        
        for token_i in range(1, self.max_l):
        
            out = self.forward(source, outputs) #[B, L, n_token]
            prob = nn.functional.softmax(out, dim=2)[:,-1] #[B, n_token]
            val, idx = torch.topk(prob, 1) # (B,1)
           
            outputs = torch.cat([outputs, idx[:,0].view(-1,1)], dim=-1) # (B,L+1)
            not_over = torch.minimum(not_over, torch.ne(outputs[:,-1], eos_id).long()) #[B]
            if torch.sum(not_over)==0: 
                break
        return outputs # (B,L)


    def _infer_beam_batch(self, source_b, top_k=1, eos_id=2, beam_size=2, mode='greedy'):
        # source: [B, S, E],
        B, S, E = source_b.shape
        
        ## initial 
        with torch.no_grad():
            outputs_b = torch.ones((B, 1), dtype=torch.long).to(source_b.device) * self.sos_id
            out_b = self.forward(source_b, outputs_b)  # B, L, n_tokens
            prob_b = F.softmax(out_b, dim=2)[:,-1] # B, n_tokens
            vals, idxs = torch.topk(prob_b, beam_size)


            beam_outputs_b = []
            beam_scores_b = []
            beam_status_b = [False]*beam_size
            for beam_id in range(beam_size):
                temp_currents = torch.cat([outputs_b, idxs[:,beam_id].view(-1,1)], dim=-1)
                temp_score = vals[:, beam_id]
                beam_scores_b.append(temp_score)
                beam_outputs_b.append(temp_currents)
            


            for _ in range(2, self.max_l):
                outputs_new_b = []
                for beam_id in range(beam_size):
                    
                    temp_out = self.forward(source_b, beam_outputs_b[beam_id])
                    prob = F.softmax(temp_out, dim=2)[:,-1]
                    vals, idxs = torch.topk(prob, beam_size)
                    
                    for b_id in range(beam_size):
                        # if beam_status[beam_id]:
                        #     continue
                        temp_score = beam_scores_b[beam_id]*vals[:, b_id]
                        temp_output = torch.cat([beam_outputs_b[beam_id], idxs[:,b_id].view(-1,1)], dim=-1)
                        outputs_new_b.append([temp_output, temp_score])
                
                
                temp_beam_outputs = []
                temp_beam_scores = []
                for id_b in range(B):
                    outputs_new = []
                    for output in outputs_new_b:
                        temp_o = output[0][id_b:id_b+1,:]
                        temp_s = output[1][id_b]
                        outputs_new.append([temp_o, temp_s.item()])
                    results =sorted(outputs_new, key=lambda x : x[1], reverse=True)[:beam_size]

                    for _, result in enumerate(results):
                        temp_beam_outputs.append(result[0])
                        temp_beam_scores.append(result[1])
                        # beam_status_b[i, id_b] = (torch.eq(result[0][:,-1], eos_id))


                
                for id_beam in range(beam_size):
                    list_temp_o = []
                    list_temp_s = []
                    for id_b in range(B):
                        temp_id = id_b*beam_size + id_beam
                        # print('id:',temp_id)
                        # print('len:',len(temp_beam_outputs))
                        temp_o = temp_beam_outputs[temp_id]
                        temp_s = temp_beam_scores[temp_id]
                        list_temp_o.append(temp_o)
                        list_temp_s.append(temp_s)


                    beam_outputs_b[id_beam] = torch.cat(list_temp_o, dim=0)
                    beam_scores_b[id_beam] = torch.from_numpy(np.array(list_temp_s)).to(device)

            list_outputs = []
            for id_b in range(B):
                beam_scores = []
                beam_outputs = []
                for id_beam in range(beam_size):
                    beam_scores.append(beam_scores_b[id_beam][id_b:id_b+1].cpu())
                    beam_outputs.append(beam_outputs_b[id_beam][id_b:id_b+1])
                good_id = np.argmax(beam_scores)
                outputs = beam_outputs[good_id]
                list_outputs.append(outputs)
        return torch.cat(list_outputs)


    def generate_square_subsequent_mask(self, sz):
        #float mask, -inf无法关注，0可以关注
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1) #1。预测i位置可以用i及之前位置的输入


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='embedding'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                try:
                    norm = torch.norm(param.grad)
                    if norm != 0 and not torch.isnan(norm):
                        r_at = epsilon * param.grad / norm
                        param.data.add_(r_at)
                except Exception as e:
                    continue

    def restore(self, emb_name='embedding'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class PGD():
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name='token_embedding', is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='token_embedding'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                try:
                    self.grad_backup[name] = param.grad.clone()
                except Exception as e:
                    continue

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                try:
                    param.grad = self.grad_backup[name]
                except Exception as e:
                    continue




if __name__ == '__main__':
    model = TranslationModel(150, 80, 1500,3, 3, 32, 4)
    source = torch.randint(0,1500, (1, 150))
    pred = model(source, beam=1)
    print('test')
