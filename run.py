import os
from utils.clip_dataloader import bert_data as weibo_data
from utils.weibo21_clip_dataloader import bert_data as weibo21_data
from model.trainer import Trainer as MMDFNDTrainer

class Run():
    def __init__(self,
                 config
                 ):
        self.configinfo = config

        self.use_cuda = config['use_cuda']
        self.model_name = config['model_name']
        self.lr = config['lr']
        self.batchsize = config['batchsize']
        self.emb_type = config['emb_type']
        self.emb_dim = config['emb_dim']
        self.max_len = config['max_len']
        self.num_workers = config['num_workers']
        self.vocab_file = config['vocab_file']
        self.early_stop = config['early_stop']
        self.bert = config['bert']
        self.root_path = config['root_path']
        self.mlp_dims = config['model']['mlp']['dims']
        self.dropout = config['model']['mlp']['dropout']
        self.seed = config['seed']
        self.weight_decay = config['weight_decay']
        self.epoch = config['epoch']
        self.save_param_dir = config['save_param_dir']
        self.dataset = config['dataset']
        self.tag = config.get('tag', 'run')
        self.output_dir = config.get('output_dir', './outputs')
        self.limit_batches = config.get('limit_batches', 0)
        self.eval_only = config.get('eval_only', False)
        self.eval_ckpt = config.get('eval_ckpt', None)
        self.amp = config.get('amp', False)
        self.optim = config.get('optim', 'adam')
        self.ada_lr = config.get('ada_lr', 1e-3)
        if config['dataset']=="weibo":
            self.root_path = './data/'
            self.train_path = self.root_path + 'train_origin.csv'
            self.val_path = self.root_path + 'val_origin.csv'
            self.test_path = self.root_path + 'test_origin.csv' 
            self.category_dict = {
                "经济": 0,
                "健康": 1,
                "军事": 2,
                "科学": 3,
                "政治": 4,
                "国际": 5,
                "教育": 6,
                "娱乐": 7,
                "社会": 8
            }
        if config['dataset']=="weibo21":
            self.root_path = './Weibo_21/'
            self.train_path = self.root_path + 'train_datasets.xlsx'#weibo21
            self.val_path = self.root_path + 'val_datasets.xlsx'#weibo21
            self.test_path = self.root_path + 'test_datasets.xlsx'#weibo21
            self.category_dict = {
                "科技": 0,
                "军事": 1,
                "教育考试": 2,
                "灾难事故": 3,
                "政治": 4,
                "医药健康": 5,
                "财经商业": 6,
                "文体娱乐": 7,
                "社会生活": 8
            }
    def get_dataloader(self,dataset):
        if self.emb_type == 'bert':
            if dataset =="weibo":
                loader = weibo_data(max_len=self.max_len, batch_size=self.batchsize, vocab_file=self.vocab_file,
                              category_dict=self.category_dict, num_workers=self.num_workers)
            if dataset =="weibo21":
                loader = weibo21_data(max_len=self.max_len, batch_size=self.batchsize, vocab_file=self.vocab_file,
                              category_dict=self.category_dict, num_workers=self.num_workers)

        if dataset =="weibo":
            train_loader = loader.load_data(self.train_path,'data/train_loader.pkl','data/train_clip_loader.pkl',True)
            val_loader = loader.load_data(self.val_path,'data/val_loader.pkl','data/val_clip_loader.pkl',False)
            test_loader = loader.load_data(self.test_path,'data/test_loader.pkl','data/test_clip_loader.pkl',False)
        if dataset =="weibo21":
            train_loader = loader.load_data(self.train_path, 'Weibo_21/train_loader.pkl', 'Weibo_21/train_clip_loader.pkl', True)
            val_loader = loader.load_data(self.val_path, 'Weibo_21/val_loader.pkl', 'Weibo_21/val_clip_loader.pkl', False)
            test_loader = loader.load_data(self.test_path, 'Weibo_21/test_loader.pkl', 'Weibo_21/test_clip_loader.pkl', False)
        return train_loader, val_loader, test_loader

    def get_smoke_loader(self):
        """Load only the test split, for fast 1-3 batch smoke runs."""
        if self.dataset == "weibo":
            loader = weibo_data(max_len=self.max_len, batch_size=self.batchsize,
                                vocab_file=self.vocab_file,
                                category_dict=self.category_dict, num_workers=0)
            return loader.load_data(self.test_path, 'data/test_loader.pkl',
                                    'data/test_clip_loader.pkl', False)
        loader = weibo21_data(max_len=self.max_len, batch_size=self.batchsize,
                              vocab_file=self.vocab_file,
                              category_dict=self.category_dict, num_workers=0)
        return loader.load_data(self.test_path, 'Weibo_21/test_loader.pkl',
                                'Weibo_21/test_clip_loader.pkl', False)

    def config2dict(self):
        config_dict = {}
        for k, v in self.configinfo.items():
            config_dict[k] = v
        return config_dict

    def main(self):
        if self.limit_batches or self.eval_only:
            train_loader = val_loader = test_loader = self.get_smoke_loader()
        else:
            train_loader, val_loader, test_loader = self.get_dataloader(self.dataset)


        if self.model_name == 'MMDFND':
            trainer = MMDFNDTrainer(emb_dim=self.emb_dim, mlp_dims=self.mlp_dims, bert=self.bert,
                                    use_cuda=self.use_cuda, lr=self.lr, train_loader=train_loader, dropout=self.dropout,
                                    weight_decay=self.weight_decay, val_loader=val_loader, test_loader=test_loader,
                                    category_dict=self.category_dict, early_stop=self.early_stop, epoches=self.epoch,
                                    tag=self.tag, output_dir=self.output_dir,
                                    limit_batches=self.limit_batches, seed=self.seed,
                                    batchsize=self.batchsize, dataset=self.dataset, max_len=self.max_len,
                                    eval_only=self.eval_only, eval_ckpt=self.eval_ckpt,
                                    amp=self.amp, optim=self.optim, ada_lr=self.ada_lr)
        trainer.train()
