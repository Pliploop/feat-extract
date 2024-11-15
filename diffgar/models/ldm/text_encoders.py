from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5EncoderModel
from torch import nn
import torch

class T5TextEncoder(nn.Module):
    
    def __init__(self, model_name = 'google-t5/t5-base'):
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        self.max_length = 77
        print('T5TextEncoder initialized')
    
    @torch.no_grad()
    def get_text_embedding(self, text, return_dict = True, return_tokenizer_only = False, **kwargs):
        text_input = self.tokenizer(text, return_tensors = 'pt', padding = True, truncation = True, max_length = self.max_length)
            
        if return_tokenizer_only:
            return text_input
        
        device = next(self.encoder.parameters()).device
            
        text_embed = self.encoder(input_ids = text_input['input_ids'].to(device), attention_mask = text_input['attention_mask'].to(device))
            
            
        if return_dict:
            text_input.update(text_embed)
        else:
            text_input = text_embed
        
        return text_input
    
    