import csv
from config import Config
conf = Config()
from torch.utils.data import DataLoader
from dataset import TranslationDataset_semi
from models import TranslationModel
from utils import to_device, Checkpoint
from tqdm import tqdm

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def array2str(arr):
    out = ''
    for i in range(len(arr)):
        if arr[i]==conf['pad_id'] or arr[i]==conf['eos_id']:
            break
        if arr[i]==conf['sos_id']:
            continue
        out += str(int(arr[i])) + ' '
    if len(out.strip())==0:
        out = '0'
    return out.strip()

def get_model():
    return TranslationModel(conf['input_l'], conf['output_l'], conf['n_token'],
                            encoder_layer=conf['n_layer'], decoder_layer=conf['n_layer'], d=conf['n_dim'], n_head=conf['n_head'])


def invoke(input_data_path, output_data_path):
    model_file = conf['model_dir']+'/model_cider.pt'
    test_data = TranslationDataset_semi(input_data_path, conf['input_l'], conf['output_l'], status='test')
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4, drop_last=False)

    model = get_model()
    checkpoint = Checkpoint(model = model)
    try:
        checkpoint.resume(model_file)
    except Exception as e:
        print(e)
    
    model.to(device) 
    model.eval()

    fp = open(output_data_path, 'w', newline='')
    writer = csv.writer(fp)
    tot = 0

    for source in tqdm(test_loader, ncols=100):
        # print('len num', len(source))
        # print('source', source)
        source = to_device(source, device)
        pred = model.forward(source, mode='beam')
        pred = pred.cpu().numpy()
        for i in range(pred.shape[0]):
            writer.writerow([tot, array2str(pred[i])])
            tot += 1
    fp.close()


    

if __name__ == '__main__':
    pass