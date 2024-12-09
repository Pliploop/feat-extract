import torch
from torch import nn
from pytorch_lightning import LightningModule

from feat_extract.models.clap.src.laion_clap import CLAP_Module
from feat_extract.models.muleT5.muleT5 import MuleT5EncoderPair
from feat_extract.models.muscall.muscall.models.muscall import MusCALL
from feat_extract.models.music2latent.music2latent_wrapper import Music2Latent

def get_encoder(encoder, encoder_kwargs=None):
    if encoder_kwargs is None:
        encoder_kwargs = {}
    return eval(encoder)(**encoder_kwargs)

class FeatureExtractor(LightningModule):
    def __init__(
        self,
        encoder='clap', ## this will be used for clap score, but the text encoder can be different
        encoder_kwargs=None,
        encoder_ckpt=None,
        freeze_encoder=True,
        device=None,
        **kwargs
        
    ):
        super().__init__()

        self.encoder = get_encoder(encoder, encoder_kwargs)
        if encoder_ckpt:
            self.encoder.load_ckpt(encoder_ckpt)
        
        if freeze_encoder:
            self.encoder.freeze()
            
        if device is not None:
            self.to(device)