import torch
from torch import nn
from pytorch_lightning import LightningModule

from feat_extract.models.clap.src.laion_clap import CLAP_Module
from feat_extract.models.muleT5.muleT5 import MuleT5EncoderPair
from feat_extract.models.muscall.muscall.models.muscall import MusCALL

def get_encoder(encoder, encoder_kwargs=None):
    return eval(encoder)(**encoder_kwargs)

class FeatureExtractor(LightningModule):
    def __init__(
        self,
        encoder='clap', ## this will be used for clap score, but the text encoder can be different
        encoder_kws=None,
        encoder_ckpt=None,
        freeze_encoder=True,
        device=None,
        **kwargs
        
    ):
        super().__init__()

        self.encoder = get_encoder(encoder, encoder_kws)
        if encoder_ckpt:
            self.encoder.load_ckpt(encoder_ckpt)
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            
        if device is not None:
            self.to(device)