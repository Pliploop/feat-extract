from music2latent import EncoderDecoder
from torch import nn
import torch

class Music2Latent(nn.Module):
    def __init__(self):
        super(Music2Latent, self).__init__()
        
        self.encoder = EncoderDecoder()
        self.gen = self.encoder.gen
        
        
        
        print("Music2Latent model initialized")
        print(self.encoder)
        
    def freeze(self):
        for param in self.encoder.gen.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def extract_features(self, x, pool_hop, extract_features = True):
        
        print(x.shape)
        
        latents =  self.encoder.encode(x,extract_features=extract_features)
        
        ## average the latents with average pooling with a hop of pool_hop
        split_latents = torch.split(latents, pool_hop, dim=-1)
        averaged_latents = torch.stack([torch.mean(latent, dim=-1) for latent in split_latents], dim=-1)
        averaged_latents = averaged_latents.squeeze(-1)
        
        return {
            'latents' : latents,
            'averaged_latents' : averaged_latents
        }