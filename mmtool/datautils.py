import torch
import PIL 
import pytorch_lightning as pl
import random
import io

class MappedDataset(torch.utils.data.Dataset):
    def __init__(self, ds, tx):
        self.ds = ds
        self.tx = tx
        
    def map(self, tx):
        return MappedDataset(self, tx=tx)   
        
    def __getitem__(self, idx):
        return self.tx(self.ds[idx])
        
    def __len__(self):
        return len(self.ds)


def split_into_sentences(desc):
    lst = desc.split('.')
    stripped = [l.strip('. ') for l in lst]
    whole = [s for s in stripped if s != '']
    return whole


class CLIPTx:
    def __init__(self, processor):
        self.processor = processor

    def preprocess_tx(self, dataset_tup):
        im = PIL.Image.open(io.BytesIO(dataset_tup['image_bytes']))
        sentences = split_into_sentences(dataset_tup['text'])
        sentence = random.choice(sentences)
        inputs = self.processor(text=[sentence], images=[im], return_tensors="pt")
        return inputs

    def pad_collate(self, input_list):
        ## list of dicts with elements 'input ids, attention mask, pixel_values'
        token_dict = {'input_ids':[d['input_ids'][0] for d in input_list],
                        'attention_mask':[d['attention_mask'][0] for d in input_list]}

        ans = self.processor.tokenizer.pad(token_dict, padding='longest', return_tensors='pt')
        ans['pixel_values'] = torch.cat([d['pixel_values'] for d in input_list])
        return ans


class MultiModalDataModule(pl.LightningDataModule):
    def __init__(self, dataset, processor, test_size, batch_size, num_workers):
        super().__init__()
        self.processor = processor
        self.dataset = dataset.shuffle().train_test_split(test_size=test_size)
        self.preproc = CLIPTx(processor)
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(MappedDataset(self.dataset['train'], self.preproc.preprocess_tx),
                                             batch_size=self.batch_size, 
                                           shuffle=True, num_workers=self.num_workers, collate_fn=self.preproc.pad_collate)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(MappedDataset(self.dataset['test'], self.preproc.preprocess_tx), 
                                            batch_size=self.batch_size, 
                                           shuffle=True, num_workers=self.num_workers, collate_fn=self.preproc.pad_collate)