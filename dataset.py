
# import json
# import glob
# import cv2
# from pathlib import Path
import torch
from torch.utils.data import Dataset
# from torchvision import transforms
# from transformers import BertTokenizer
import numpy as np
# from PIL import Image
import random
import time
import csv
import traceback

class BaseDataset(Dataset):
    def _try_getitem(self, idx):
        raise NotImplementedError
    def __getitem__(self, idx):
        # wait = 0.1
        while True:
            try:
                ret = self._try_getitem(idx)
                return ret
            except KeyboardInterrupt:
                break
            except (Exception, BaseException) as e: 
                exstr = traceback.format_exc()
                print(exstr)
                print('read error, waiting:', wait)
                time.sleep(wait)
                wait = min(wait*2, 1000)

class TranslationDataset(BaseDataset):
    def __init__(self, data_file, input_l, output_l, sos_id=1, eos_id=2, pad_id=0, status='valid'):
        with open(data_file, 'r') as fp:
            reader = csv.reader(fp)
            self.samples = [row for row in reader]
            self.input_l = input_l
            self.output_l = output_l
            self.sos_id = sos_id
            self.pad_id = pad_id
            self.eos_id = eos_id
            self.status = status
    def __len__(self):
        return len(self.samples)
    def _try_getitem(self, idx):
        temp_ratio = 0.15
        source = [int(x) for x in self.samples[idx][1].split()]
        if self.status == 'train':
            if random.random() < 0.25:
                source = random_del_seq(source, temp_ratio)
            if random.random() < 0.25:
                source, _ = random_mask_seq_m(source, 3, temp_ratio)
            if random.random() < 0.25:
                source, _ = random_erode_seq(source, temp_ratio)
            if random.random() < 0.25:
                source = random_swap_seq(source, temp_ratio)

            # list_source_aug = [source]
            # temp = random_del_seq(source, ratio=0.1)
            # list_source_aug.append(temp)
            # temp, _ = random_mask_seq_m(source, 3, temp_ratio, 4)
            # list_source_aug.append(temp)
            # # temp, _ = random_erode_seq(source, temp_ratio)
            # # list_source_aug.append(temp)
            # # temp = random_swap_seq(source, temp_ratio)
            # # list_source_aug.append(temp)
            # source = random.choice(list_source_aug)

        if len(source)<self.input_l:
            source.extend([self.pad_id] * (self.input_l-len(source)))
        if len(self.samples[idx])<3:
            return np.array(source)[:self.input_l]
        target = [self.sos_id] + [int(x) for x in self.samples[idx][2].split()] + [self.eos_id]
        if len(target)<self.output_l:
            target.extend([self.pad_id] * (self.output_l-len(target)))
        return np.array(source)[:self.input_l], np.array(target)[:self.output_l]

def random_mask_seq(source, mask_id=3, mask_ratio=0.2):
    source_new = []
    mask_target = []
    for s in source:
        if random.random() < mask_ratio:
            temp_r = random.random()
            if temp_r < 0.8:
                source_new.append(mask_id)
                mask_target.append(s)
            elif temp_r < 0.9:
                source_new.append(random.randint(5, 1299))
                mask_target.append(s)
            else:
                source_new.append(s)
                mask_target.append(-100)
        else:
            source_new.append(s)
            mask_target.append(-100)
    return source_new, mask_target
    
# def random_del_seq(source, ratio=0.1):
#     len_ori = len(source)
#     source_new = []
#     for s in source:
#         if random.random() < ratio:
#             source_new.append(random.randint(3, 1499))
#         else:
#             source_new.append(s)
#     source_new.extend([0]*(len_ori-len(source_new)))
#     return source_new

def random_del_seq(source, ratio=0.1):
    len_ori = len(source)
    source_new = []
    for s in source:
        if random.random() < ratio:
            continue
        else:
            source_new.append(s)
    source_new.extend([0]*(len_ori-len(source_new)))
    return source_new

def random_swap_seq(source, ratio=0.2):
    source_new = []
    for i, s in enumerate(source):
        if len(source_new) > i:
            continue
        if random.random() < ratio and i>1:
            try:
                source_new.append(source[i+1])
                source_new.append(s)
            except Exception as e:
                source_new.append(s)
                continue
        else:
            source_new.append(s)
    return source_new   

def random_mask_seq_m(source, mask_id=3, mask_ratio=0.2, mask_len=4):
    source_new = []
    mask_target = []
    for i, s in enumerate(source):
        if len(source_new) > i:
            continue
        if random.random() < mask_ratio:
            mask_size = random.randint(2, mask_len)
            if mask_size>0:
                for _ in range(mask_size):
                    if len(source_new)<len(source):
                        source_new.append(mask_id)
                        mask_target.append(s)
            else:
                source_new.append(s)
                mask_target.append(-100)
        else:
            source_new.append(s)
            mask_target.append(-100)
    return source_new, mask_target

def random_inver_seq(source, ratio=0.5):
    if random.random() < ratio:
        source_new = source[::-1]
    else:
        source_new = source
    return source_new

def random_erode_seq(source, ratio=0.2):
    pass
    source_new = []
    mask_target = []
    for i, s in enumerate(source):
        if len(source_new) > i:
            continue
        if random.random() < ratio:
            erode_size = random.randint(2, 4)
            temp_id = source[i]
            for _ in range(erode_size):
                source_new.append(temp_id)
                mask_target.append(temp_id)
        else:
            source_new.append(s)
            mask_target.append(-100)
    return source_new, mask_target



from config import Config
conf = Config()
pre_ratio = conf['train_ratio']


class TranslationDataset_mt(BaseDataset):
    def __init__(self, data_file, input_l, output_l, sos_id=1, eos_id=2, pad_id=0, status='valid'):
        with open(data_file, 'r') as fp:
            reader = csv.reader(fp)
            self.samples = [row for row in reader]
            self.input_l = input_l
            self.output_l = output_l
            self.sos_id = sos_id
            self.pad_id = pad_id
            self.eos_id = eos_id
            self.status = status
    def __len__(self):
        return len(self.samples)
    def _try_getitem(self, idx):
        temp_ratio = 0.3
        source = [int(x) for x in self.samples[idx][1].split()]
        if self.status == 'train':
            # if random.random() < 0.25:
            #     source = random_del_seq(source, temp_ratio)
            # if random.random() < 0.25:
            #     source, _ = random_mask_seq_m(source, 3, temp_ratio)
            # if random.random() < 0.25:
            #     source, _ = random_erode_seq(source, temp_ratio)
            # if random.random() < 0.25:
            #     source = random_swap_seq(source, temp_ratio)

            list_source_aug = [source]
            temp = random_del_seq(source, ratio=0.1)
            list_source_aug.append(temp)
            temp, _ = random_mask_seq_m(source, 3, temp_ratio, 4)
            list_source_aug.append(temp)
            temp, _ = random_erode_seq(source, temp_ratio)
            list_source_aug.append(temp)
            temp = random_swap_seq(source, temp_ratio)
            list_source_aug.append(temp)
            source = random.choice(list_source_aug)

        if len(source)<self.input_l:
            source.extend([self.pad_id] * (self.input_l-len(source)))
        if len(self.samples[idx])<3:
            return np.array(source)[:self.input_l]
        target = [self.sos_id] + [int(x) for x in self.samples[idx][2].split()] + [self.eos_id]
        if len(target)<self.output_l:
            target.extend([self.pad_id] * (self.output_l-len(target)))
        return np.array(source)[:self.input_l], np.array(target)[:self.output_l]
        return  np.array(source)[:self.input_l], np.array(target)[:self.output_l], \
                np.array(source_new_mask)[:self.input_l], np.array(mask_target)[:self.input_l], \
                np.array(source_new_del)[:self.input_l],\
                np.array(target_new_mask)[:self.input_l], np.array(target_mask_target)[:self.output_l]




class TranslationDataset_pretrain(BaseDataset):
    def __init__(self, data_file, input_l, sos_id=1, eos_id=2, pad_id=0, mask_id=3, sep_id=4):
        with open(data_file, 'r') as fp:
            reader = csv.reader(fp)
            self.samples = [row for row in reader]
            self.input_l = input_l
            # self.output_l = output_l
            self.sos_id = sos_id
            self.pad_id = pad_id
            self.eos_id = eos_id
            self.mask_id = mask_id
            self.sep_id = sep_id
    def __len__(self):
        return len(self.samples)
    def _try_getitem(self, idx):

        source = [int(x) for x in self.samples[idx][0].split()]
        source_de = [self.sos_id] + [int(x) for x in self.samples[idx][0].split()] + [self.eos_id]
        
        if len(source)<self.input_l:
            source.extend([self.pad_id] * (self.input_l-len(source)))
        
        if len(source_de)<self.input_l:
            source_de.extend([self.pad_id] * (self.input_l-len(source_de)))
        
        source_mask_sim, mask_target_sim = random_mask_seq(source, mask_id=self.mask_id, mask_ratio=pre_ratio)
        source_mask, mask_target = random_mask_seq_m(source, mask_id=self.mask_id, mask_ratio=pre_ratio, mask_len=4)
        source_erd, _ = random_erode_seq(source)
        source_swp = random_swap_seq(source, ratio=pre_ratio)
        source_inv = random_inver_seq(source)

        source_mask_de, mask_target_de = random_mask_seq_m(source_de, mask_id=self.mask_id, mask_ratio=pre_ratio, mask_len=4)
        source_swp_de = random_swap_seq(source_de, ratio=pre_ratio)


        

        return np.array(source)[:self.input_l], \
                np.array(source_mask_sim)[:self.input_l], np.array(mask_target_sim)[:self.input_l], \
                np.array(source_mask)[:self.input_l], np.array(mask_target)[:self.input_l], \
                np.array(source_erd)[:self.input_l],  np.array(source_swp)[:self.input_l], \
                np.array(source_mask_de)[:self.input_l],  np.array(mask_target_de)[:self.input_l], \
                np.array(source_swp_de)[:self.input_l], np.array(source_inv)[:self.input_l],\

    

class TranslationDataset_pretrain_semi(BaseDataset):
    def __init__(self, data_file, input_l, sos_id=1, eos_id=2, pad_id=0, mask_id=3, sep_id=4):
        with open(data_file, 'r') as fp:
            reader = csv.reader(fp)
            self.samples = []
            for row in reader:
                    self.samples.append(row)
            # self.samples = [row for row in reader]
            self.input_l = input_l
            # self.output_l = output_l
            self.sos_id = sos_id
            self.pad_id = pad_id
            self.eos_id = eos_id
            self.mask_id = mask_id
            self.sep_id = sep_id
    def __len__(self):
        return len(self.samples)
    def _try_getitem(self, idx):
        source = []
        sps = self.samples[idx]
        if len(sps)>1:
            sp_all = sps[0] + ' ' + str(self.sep_id)+ ' '  + sps[1]
        else:
            sp_all = sps[0] + ' ' + str(self.sep_id)+ ' '  + sps[0]

        for x in sp_all.split():
            source.append(int(x))

        source_de = [self.sos_id] + [int(x) for x in self.samples[idx][0].split()] + [self.eos_id]
        
        if len(source)<self.input_l:
            source.extend([self.pad_id] * (self.input_l-len(source)))
        
        if len(source_de)<self.input_l:
            source_de.extend([self.pad_id] * (self.input_l-len(source_de)))
        
        source_mask_sim, mask_target_sim = random_mask_seq(source, mask_id=self.mask_id, mask_ratio=pre_ratio)
        source_mask, mask_target = random_mask_seq_m(source, mask_id=self.mask_id, mask_ratio=pre_ratio, mask_len=4)
        source_erd, _ = random_erode_seq(source)
        source_swp = random_swap_seq(source, ratio=pre_ratio)
        source_inv = random_inver_seq(source)

        source_mask_de, mask_target_de = random_mask_seq_m(source_de, mask_id=self.mask_id, mask_ratio=pre_ratio, mask_len=4)
        source_swp_de = random_swap_seq(source_de, ratio=pre_ratio)


        

        return np.array(source)[:self.input_l], \
                np.array(source_mask_sim)[:self.input_l], np.array(mask_target_sim)[:self.input_l], \
                np.array(source_mask)[:self.input_l], np.array(mask_target)[:self.input_l], \
                np.array(source_erd)[:self.input_l],  np.array(source_swp)[:self.input_l], \
                np.array(source_mask_de)[:self.input_l],  np.array(mask_target_de)[:self.input_l], \
                np.array(source_swp_de)[:self.input_l], np.array(source_inv)[:self.input_l],\



class TranslationDataset_semi(BaseDataset):
    def __init__(self, data_file, input_l, output_l, sos_id=1, eos_id=2, pad_id=0, sep_id=4, status='valid'):
        with open(data_file, 'r') as fp:
            reader = csv.reader(fp)
            self.samples = []
            for row in reader:
                self.samples.append(row)
            # self.samples = self.samples[:1024]
            self.input_l = input_l
            self.output_l = output_l
            self.sos_id = sos_id
            self.pad_id = pad_id
            self.eos_id = eos_id
            self.status = status
            self.sep_id = sep_id
    def __len__(self):
        return len(self.samples)
    def _try_getitem(self, idx):
        temp_ratio = 0.15

        source = []
        sps = self.samples[idx]
        if self.status == 'train':
            if len(sps)>3:
                if random.random() < 0.5:
                    sp_all = sps[1] + ' ' + str(self.sep_id)+ ' '  + sps[3]
                else:
                    sp_all = sps[1] + ' ' + str(self.sep_id) + ' ' + str(self.pad_id)
            else:
                if random.random() < 0.5:
                    sp_all = sps[1] + ' ' + str(self.sep_id) + ' '  + sps[1]
                else:
                    sp_all = sps[1] + ' ' + str(self.sep_id) + ' ' + str(self.pad_id)
        else:
            if len(sps)>3:
                sp_all = sps[1] + ' ' + str(self.sep_id)+ ' '  + sps[3]
            else:
                sp_all = sps[1] + ' ' + str(self.sep_id) + ' '  + sps[1]


        for x in sp_all.split():
            source.append(int(x))

        if self.status == 'train':
            source_t = source.copy()
            # if random.random() < 0.25:
            #     source = random_del_seq(source, temp_ratio)
            if random.random() < 0.15:
                source, _ = random_mask_seq_m(source, 3, temp_ratio)
            if random.random() < 0.15:
                source, _ = random_erode_seq(source, temp_ratio)
            if random.random() < 0.15:
                source = random_swap_seq(source, temp_ratio)

            # list_source_aug = [source]
            # temp = random_del_seq(source, ratio=0.1)
            # list_source_aug.append(temp)
            # temp, _ = random_mask_seq_m(source, 3, temp_ratio, 4)
            # list_source_aug.append(temp)
            # # temp, _ = random_erode_seq(source, temp_ratio)
            # # list_source_aug.append(temp)
            # # temp = random_swap_seq(source, temp_ratio)
            # # list_source_aug.append(temp)
            # source = random.choice(list_source_aug)

        if len(source)<self.input_l:
            source.extend([self.pad_id] * (self.input_l-len(source)))
        if self.status == 'train':
            source_t.extend([self.pad_id]*(self.input_l-len(source_t)))
        
        if self.status == 'test':
            s = np.array(source)[:self.input_l]
            return s
        
        # if len(self.samples[idx])<4:
        #     return np.array(source)[:self.input_l]
        
        target = [self.sos_id] + [int(x) for x in self.samples[idx][2].split()] + [self.eos_id]
        if len(target)<self.output_l:
            target.extend([self.pad_id] * (self.output_l-len(target)))

        s = np.array(source)[:self.input_l]
        t = np.array(target)[:self.output_l]

        if self.status == 'train':
            s_t = np.array(source_t)[:self.input_l]
            # print(s.shape, t.shape)
            return s, t, s_t
        else:
            return s, t

