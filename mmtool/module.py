import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
import math
from transformers import CLIPModel
import transformers
import torch

class ContrastLoss(nn.Module):
    def __init__(self, tau):
        super().__init__()
        self.tau = tau
    
    def loss(self, logits):
        target = torch.arange(logits.shape[0]).long().to(logits.device)
        return F.cross_entropy(logits, target, reduction='mean')
    
    def forward(self, vecs1, vecs2):
        assert vecs1.shape == vecs2.shape
        # cosine similarity as logits
        logits = (math.exp(self.tau) * vecs1) @ vecs2.t()
        return self.loss(logits) + self.loss(logits.t())

# see https://huggingface.co/transformers/_modules/transformers/models/clip/modeling_clip.html#CLIPModel
def warmup(warmup_epochs, k=1.):
    def fun(epoch):
        if epoch < warmup_epochs:
            return epoch/warmup_epochs #0
        
        curr_epoch = max(epoch - warmup_epochs,1) # >= 1
        return 1./math.sqrt(max(curr_epoch/k, 1.))
    
    return fun

class CLIPModule(pl.LightningModule):
    def __init__(self, path_or_model_name, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        model = CLIPModel.from_pretrained(path_or_model_name)
        model.logit_scale = nn.Parameter(torch.tensor(math.log(self.hparams.init_scale), 
                                  device=model.logit_scale.device,
                                  dtype=model.logit_scale.dtype))
        
        self.model = model

        
    def forward(self, inputs):
        return  self.model(**inputs, return_loss=True)

    def basic_step(self, batch, train_or_val):
        batch_out = self.forward(batch)
        loss = batch_out.loss
        self.log(f'{train_or_val}_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        return self.basic_step(batch, 'val')
    
    def training_step(self, batch, batch_idx):
        self.log('logit_scale', self.model.logit_scale)
        ans =  self.basic_step(batch, 'train')
        return ans
    
    def configure_optimizers(self):
        m = self.model
        h = self.hparams
        
        
        optimizer = transformers.AdamW([ {'name':'logit_scale',
                                         'params':[m.logit_scale], 
                                         'lr':h.logit_scale_lr, 
                                         'weight_decay':h.logit_scale_decay}
                                       ],
                                       lr=h.lr, weight_decay=h.decay)
        
        names = ['text_model','text_projection','vision_model','visual_projection']
        
        for name in names:
            lr = h[name + '_lr']
            decay = h[name + '_decay']

            submod = m.get_submodule(name)
            
            bias_params = []
            non_bias_params = []
            for (nm, p) in submod.named_parameters():
                if nm.endswith('bias'):
                    bias_params.append(p)
                else:
                    non_bias_params.append(p)
                    
            
            group_no_bias = {'name':name, 'params':non_bias_params, 'lr':lr, 'weight_decay':decay}
            group_bias = {'name':name +'_biases', 'params':bias_params, 'lr':lr, 'weight_decay':0.}
            optimizer.add_param_group(group_no_bias)
            optimizer.add_param_group(group_bias)

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup(h.warmup_epochs))
        return {'optimizer':optimizer, 'lr_scheduler':lr_scheduler}
    

    