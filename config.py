
class Config(dict):
    def version_config(self, version):
        hp = {
            1: {'n_epoch':150,  'batch':32,  'valid_batch':32, 'n_layer':3, 'n_epoch_pre':80},
            2: {'n_epoch':150,  'batch':32, 'valid_batch':32, 'n_layer':6, 'n_epoch_pre':80},
            3: {'n_epoch':150, 'batch':16, 'valid_batch':16, 'n_layer':6, 'n_head':16, 'n_dim':1024, 'n_epoch_pre':80},
            4: {'n_epoch':50,  'batch':32, 'valid_batch':8,  'n_layer':6, 'n_head':16, 'n_dim':512},
            5: {'n_epoch':60,  'batch':32,  'valid_batch':32, 'n_layer':3, 'n_epoch_pre':40, 'model_dir': './checkpoint/%d_pretrain'%version },
            6: {'n_epoch':120,  'batch':32,  'valid_batch':32, 'n_layer':3, 'n_epoch_pre':80, 
                'model_dir': './checkpoint/%d_pretrain'%version , 
                'pre_model_dir': './checkpoint/6_pretrain/model_pretrain.pt'},
            7: {'n_epoch':120,  'batch':8,  'valid_batch':32, 'n_layer':3, 'n_epoch_pre':60, 'model_dir': './checkpoint/%d_pretrain'%version , 'pre_model_dir': './checkpoint/7_pretrain/model_pretrain.pt'},
            8: {'n_epoch':120,  'batch':8,  'valid_batch':32, 'n_layer':3, 'n_epoch_pre':60, 'model_dir': './checkpoint/%d_pretrain'%version , 'pre_model_dir': './checkpoint/6_pretrain/model_pretrain.pt'},
            9: {'n_epoch':80,  'batch':16,  'valid_batch':8, 'n_layer':6, 'n_epoch_pre':60, 
                'model_dir': './checkpoint/%d_pretrain'%version , 
                'pre_model_dir': './checkpoint/9_pretrain/model_pretrain.pt'},
            10: {'n_epoch':100,  'batch':16,  'valid_batch':8, 'n_layer':6, 'n_epoch_pre':80, 
                 'model_dir': './checkpoint/%d_pretrain'%version , 
                 'pre_model_dir': './checkpoint/%d_pretrain/model_pretrain.pt'%version},
            11: {'n_epoch':100,  'batch':16,  'valid_batch':8, 'n_layer':6, 'n_epoch_pre':80, 
                 'model_dir': './checkpoint/%d_pretrain'%version , 
                 'pre_model_dir': './checkpoint/%d_pretrain/model_pretrain.pt'%version},
            12: {'n_epoch':100,  'batch':16,  'valid_batch':8, 'n_epoch_pre':80, 
                 'train_ratio':0.3,'n_layer':6, 'n_head':8, 'n_dim':512,
                 'model_dir': './checkpoint/%d_pretrain'%version , 
                 'pre_model_dir': './checkpoint/%d_pretrain/model_pretrain.pt'%version},
            13: {'n_epoch':150,  'batch':32,  'valid_batch':32, 'n_epoch_pre':80, 
                 'train_ratio':0.15, 'n_layer':12, 'n_head':8, 'n_dim':128,
                 'model_dir': './checkpoint/%d_pretrain'%version , 
                 'pre_model_dir': './checkpoint/%d_pretrain/model_pretrain.pt'%version},
            14: {'n_epoch':100,  'batch':40,  'valid_batch':64, 'n_epoch_pre':40, 
                 'train_ratio':0.15, 'n_layer':12, 'n_head':8, 'n_dim':128,
                 'dropout_en':0.1, 'dropout_de':0.1, 'dropout':0.0, 
                 'model_dir': './checkpoint/%d_pretrain'%version , 
                 'pre_model_dir': './checkpoint/%d_pretrain/model_pretrain.pt'%version},
            15: {'n_epoch':100,  'batch':32,  'valid_batch':64, 'n_epoch_pre':40, 
                 'train_ratio':0.15, 'n_layer':12, 'n_head':8, 'n_dim':128,
                 'dropout_en':0.2, 'dropout_de':0.2, 'dropout':0.0, 
                 'model_dir': './checkpoint/%d_pretrain'%version , 
                 'pre_model_dir': './checkpoint/%d_pretrain/model_pretrain.pt'%version},
            16: {'n_epoch':100,  'batch':16,  'valid_batch':64, 'n_epoch_pre':30, 
                 'train_ratio':0.2, 'n_layer':12, 'n_head':8, 'n_dim':256, 'dropout':0.2, 
                 'en_de_ratio':0.5,
                 'model_dir': './checkpoint/%d_pretrain'%version , 
                 'pre_model_dir': './checkpoint/%d_pretrain/model_pretrain.pt'%version},
          
            17: {'n_epoch':100,  'batch':16,  'valid_batch':20, 'n_epoch_pre':20, 
                 'train_ratio':0.15, 'n_layer':6, 'n_head':16, 'n_dim':1024,
                 'dropout_en':0.0, 'dropout_de':0.0, 'dropout':0.0, 
                 'dim_ff':4096,
                 'en_de_ratio':1.0,
                 'model_dir': './checkpoint/%d_pretrain'%version , 
                 'pre_model_dir': './checkpoint/%d_pretrain/model_pretrain.pt'%version},

          

        }

          # "d_model": 1024,s
          # "decoder_attention_heads": 16,
          # "decoder_ffn_dim": 4096,
          # "decoder_layerdrop": 0.0,
          # "decoder_layers": 12,
          # "decoder_start_token_id": 2,
          # "dropout": 0.1,
          # "early_stopping": true,
          # "encoder_attention_heads": 16,
          # "encoder_ffn_dim": 4096,
          # "encoder_layerdrop": 0.0,
          # "encoder_layers": 12,

        self['n_epoch'] = hp[version].get('n_epoch', 50)
        self['n_epoch_pre'] = hp[version].get('n_epoch_pre', 80)
        self['n_layer'] = hp[version].get('n_layer', 6)
        self['batch'] = hp[version].get('batch', 8)
        self['valid_batch'] = hp[version].get('valid_batch', 8)
        self['n_head'] = hp[version].get('n_head', 8)
        self['n_dim'] = hp[version].get('n_dim', 512)
        self['dim_ff'] = hp[version].get('dim_ff', 2048)
        self['w_g'] = 1
        self['model_dir'] =  hp[version].get('model_dir', './checkpoint/%d'%version)
        self['pre_model_dir'] = hp[version].get('pre_model_dir', None)
        self['train_ratio'] = hp[version].get('train_ratio', 0.15)
        self['dropout'] = hp[version].get('dropout', 0.0)
        self['dropout_en'] = hp[version].get('dropout_en', 0.1)
        self['dropout_de'] = hp[version].get('dropout_de', 0.1)
        self['en_de_ratio'] = hp[version].get('en_de_ratio', 1.0)

        #请自己造训练测试集
        self['train_file'] = 'data/train_s.csv'
        self['valid_file'] = 'data/valid_s.csv'
        self['test_file'] = 'data/preliminary_b_test.csv'
        self['pretrain_file'] = 'data/pretrain.csv'
    
        self['input_l'] = 150
        self['output_l'] = 80
        self['n_token'] = 1650
        self['sos_id'] = 1
        self['eos_id'] = 2
        self['pad_id'] = 0
        self['sep_id'] = 3
        
    def __init__(self, version=14, seed=0):
        print('config version:', version)
        self['lr_pre'] = 1e-4
        self['lr'] = 1e-4
        
        if seed>0:
            self['model_dir'] += '_%d'%seed
        self['output_dir'] = './outputs/%d'%version
        
        self.version_config(version)
        