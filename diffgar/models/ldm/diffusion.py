from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffgar.utils.subject_processing import *
from pytorch_lightning import LightningModule

from diffusers.utils.torch_utils import randn_tensor
from diffusers import DDPMScheduler
from .unet import UNet
from pytorch_lightning.cli import OptimizerCallable
# import optim
from torch import optim
from diffgar.models.utils.schedulers import *
from .text_encoders import T5TextEncoder

import yaml



from diffgar.models.clap.src.laion_clap import CLAP_Module

def get_encoder_pair(encoder_pair, encoder_pair_kwargs=None):
    
    if encoder_pair == "clap":
        return CLAP_Module(**encoder_pair_kwargs)
    elif encoder_pair == "muscall":
        return MusCALL_Module(**encoder_pair_kwargs)
    
def get_text_encoder(text_encoder, text_encoder_kwargs=None):
    if text_encoder == 'T5':
        return T5TextEncoder(**text_encoder_kwargs)
    
    
class Slider(nn.Module):
    # simple nn parameter init with an embedding dimension
    def __init__(self, dim):
        super().__init__()
        self.slider = nn.Parameter(torch.randn(dim), requires_grad=True)
        
    def freeze(self):
        self.slider.requires_grad = False
        
    def apply(self,sequence,mask, scale = 1.0):
        
        if not mask:
            print("No mask provided, returning sequence")
            return sequence
        
        assert sequence.shape[-1] == self.slider.shape[0], "Sequence and slider dimensions do not match"
        
        return sequence + scale * self.slider * mask.unsqueeze(-1)
    

class DiffGarLDM(nn.Module):
    def __init__(
        self,
        encoder_pair='clap', ## this will be used for clap score, but the text encoder can be different
        encoder_pair_kwargs=None,
        encoder_pair_ckpt=None,
        text_encoder = None,
        text_encoder_kwargs = None,
        text_encoder_ckpt = None,
        scheduler_name='stabilityai/stable-diffusion-2-1',
        scheduler_pred_type='epsilon',
        unet_model_config=None,
        unet_ckpt=None,
        snr_gamma=None,
        freeze_encoder_pair=True,
        freeze_unet=False,
        uncondition=False,
        subject_flagging = False,
        train_slider_concept = None,
        device=None,
        **kwargs
        
    ):
        super().__init__()

        self.scheduler_name = scheduler_name
        self.unet_model_config = unet_model_config
        self.snr_gamma = snr_gamma
        self.uncondition = uncondition

        # https://huggingface.co/docs/diffusers/v0.14.0/en/api/schedulers/overview
        
        self.noise_scheduler = DDPMScheduler.from_pretrained(self.scheduler_name, subfolder="scheduler")
        self.inference_scheduler = DDPMScheduler.from_pretrained(self.scheduler_name, subfolder="scheduler")
        
        #change prediction type
        self.noise_scheduler.config.prediction_type = scheduler_pred_type
        self.inference_scheduler.config.prediction_type = scheduler_pred_type
        self.max_length = self.unet_model_config['embedding_max_length']


        print(f"Diffusion model initialized with scheduler {self.noise_scheduler}")


        # assert self.unet_model_config is not None, "UNet model config is required"
        
        self.unet = UNet.from_config(self.unet_model_config) if self.unet_model_config is not None else None
        self.set_from = "random"
        print("UNet initialized randomly.")

        self.encoder_pair = get_encoder_pair(encoder_pair, encoder_pair_kwargs)
        if encoder_pair_ckpt:
            self.encoder_pair.load_ckpt(encoder_pair_ckpt)
        self.text_encoder = self.encoder_pair
        
        if text_encoder is not None:
            self.text_encoder = get_text_encoder(text_encoder, text_encoder_kwargs)
            self.text_encoder.max_length = self.max_length
            if text_encoder_ckpt:
                self.text_encoder.load_ckpt(text_encoder_ckpt)
        
        if freeze_encoder_pair:
            self.freeze_encoder_pair()
            self.freeze_text_encoder()
            
        self.freeze_encoders = freeze_encoder_pair
        
        if unet_ckpt:
            ckpt = torch.load(unet_ckpt)
            self.unet.load_state_dict(ckpt['state_dict'])
        if freeze_unet:
            self.freeze_unet()
            
        
            
        self.subject_flagging = subject_flagging
        
        self.train_slider_concept = train_slider_concept
        
        self.first_run = False
        
        
        with torch.no_grad():
            text_dim = self.text_encoder.get_text_embedding("test")['last_hidden_state'].shape[-1]
        
        print(f"Text dimension: {text_dim}")
        
        if self.train_slider_concept:
            self.slider = Slider(text_dim)
            
            
        if device is not None:
            self.to(device)
            
    @classmethod
    def from_config(cls, config, device=None):
        config.update(device=device)
        return cls(**config)
    
    @classmethod
    def from_yaml(cls, yaml_path, device=None):
        
        if 's3://' in yaml_path:
            import s3fs
            fs = s3fs.S3FileSystem()
            with fs.open(yaml_path, "r") as file:
                config = yaml.safe_load(file)
                config = config.get('model', config)
        else:
            with open(yaml_path, "r") as file:
                config = yaml.safe_load(file)
                config = config.get('model', config) #for loading from LightningCLI configs or nondescript yaaml files
        return cls.from_config(config, device=device)
    
    @classmethod
    def from_pretrained(cls, yaml_or_config, ckpt_path, device=None):
        if isinstance(yaml_or_config, str):
            model = cls.from_yaml(yaml_or_config, device=device)
        else:
            model = cls.from_config(yaml_or_config, device=device)
            
            
        if 's3://' in ckpt_path:
            from s3torchconnector import S3Checkpoint
            checkpoint= S3Checkpoint(region='us-east-1')
            with checkpoint.reader(ckpt_path) as f:
                ckpt = torch.load(f, map_location=device)
                model.load_state_dict(ckpt['state_dict'], strict=True)
                print(f"Model loaded from {ckpt_path}")
        else:
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['state_dict'], strict=True)
            print(f"Model loaded from {ckpt_path}")
        
        return model

    def compute_snr(self, timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = self.noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr
    

        

    def encode_text(self, prompt, subject_masks = None):
        
        
        if self.subject_flagging:
            offsets = self.text_encoder.get_text_embedding(prompt,return_dict = True, return_tokenizer_only = True)['offset_mapping']
            prompt, subject_masks = create_binary_mask_from_list(prompt, offsets)

        if self.freeze_encoders:
            with torch.no_grad():
                encoded_text_dict = self.text_encoder.get_text_embedding(prompt, use_tensor = True, return_dict = True)
        else:
            encoded_text_dict = self.text_encoder.get_text_embedding(prompt, use_tensor = True, return_dict = True)
            
        if subject_masks is not None:
            encoded_text_dict['subject_masks'] = subject_masks

        return encoded_text_dict
            
            
    def freeze_encoder_pair(self):
        for param in self.encoder_pair.parameters():
            param.requires_grad = False
            
    def unfreeze_encoder_pair(self):
        for param in self.encoder_pair.parameters():
            param.requires_grad = True
            
    def freeze_unet(self):
        for param in self.unet.parameters():
            param.requires_grad = False
            
    def unfreeze_unet(self):
        for param in self.unet.parameters():
            param.requires_grad = True
            
    def freeze_text_encoder(self):
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            
    def unfreeze_text_encoder(self):
        for param in self.text_encoder.parameters():
            param.requires_grad = True
            
    def freeze(self):
        self.freeze_encoder_pair()
        self.freeze_unet()
        self.freeze_text_encoder()
        
        for param in self.parameters():
            param.requires_grad = False
            
    def unfreeze(self):
        self.unfreeze_encoder_pair()
        self.unfreeze_unet()
        self.unfreeze_text_encoder()
        
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, latents, prompt, validation_mode=False):
        
        
        bsz = latents.shape[0]
        
        device = next(self.parameters()).device
        num_train_timesteps = self.noise_scheduler.num_train_timesteps
        self.noise_scheduler.set_timesteps(num_train_timesteps, device=device)
        text_dict = self.encode_text(prompt)
        
        encoder_hidden_states = text_dict['last_hidden_state']
        boolean_encoder_mask = text_dict['attention_mask']
        subject_mask = text_dict.get('subject_masks', None)
        
        
        
        if self.train_slider_concept and self.slider and self.subject_mask:
            # randomly sample a tensor of scales for the slider of shape bsz, between -1 and 1
            scales = torch.rand(bsz) * 2 - 1
            encoder_hidden_states = self.slider.apply(encoder_hidden_states, subject_mask, scale = scales)
        
        # if self.uncondition:
        #     mask_indices = [k for k in range(len(prompt)) if random.random() < 0.1]
        #     if len(mask_indices) > 0:
        #         encoder_hidden_states[mask_indices] = 0
        #the above is cfg masking, not sure if it is needed here


        if validation_mode:
            timesteps = (self.noise_scheduler.num_train_timesteps//2) * torch.ones((bsz,), dtype=torch.int64, device=device)
        else:
            # Sample a random timestep for each instance
            timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bsz,), device=device)
        timesteps = timesteps.long()
        
        
        noise = torch.randn_like(latents)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        elif self.noise_scheduler.config.prediction_type == "sample":
            target = latents
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")


        bsz, length, device = *encoder_hidden_states.shape[0:2], encoder_hidden_states.device
        
        assert latents.shape == noisy_latents.shape, "Latents and noisy latents shape mismatch"
        assert latents.shape == target.shape, "Latents and target shape mismatch"
        
        if self.unet_model_config['classifier_free_guidance_strength'] is not None:
            guidance_scale = self.unet_model_config['classifier_free_guidance_strength']
        else:
            guidance_scale = None
        
        if self.set_from == "random":
            model_pred = self.unet(
                noisy_latents, time = timesteps, embedding = encoder_hidden_states, embedding_mask_proba = guidance_scale, embedding_scale = 1.0
            )

        if self.snr_gamma is None:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Adaptef from huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
            snr = self.compute_snr(timesteps)
            mse_loss_weights = (
                torch.stack([snr, self.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
            )
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

        assert model_pred.shape == target.shape, "Model prediction and target shape mismatch"

        return loss

    @torch.no_grad()
    def inference(self, prompt, inference_scheduler, num_steps=20, guidance_scale=None, num_samples_per_prompt=1, 
                  disable_progress=True, slider = None, slider_scale = 0):
        device = next(self.parameters()).device
        batch_size = len(prompt) * num_samples_per_prompt

        # if classifier_free_guidance:
        #     encoded_text,uncond_encoded_text = self.encode_text_classifier_free(prompt, num_samples_per_prompt)
            
        #     prompt_embeds, boolean_prompt_mask, subject_mask = encoded_text['last_hidden_state'], encoded_text['attention_mask'], encoded_text.get('subject_masks', None)
        #     # uncond_prompt_embeds, uncond_boolean_prompt_mask, _ = uncond_encoded_text['last_hidden_state'], uncond_encoded_text['attention_mask'], uncond_encoded_text.get('subject_masks', None)
            
        #     if slider and subject_mask:
        #         prompt_embeds = slider.apply(prompt_embeds,boolean_prompt_mask, scale = slider_scale)
        #         # uncond_prompt_embeds = slider.apply(uncond_prompt_embeds,uncond_boolean_prompt_mask, scale = slider_scale)
                
        #     prompt_embeds = prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
        #     boolean_prompt_mask = boolean_prompt_mask.repeat_interleave(num_samples_per_prompt, 0)
        #     uncond_prompt_embeds = uncond_prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
        #     uncond_boolean_prompt_mask = uncond_boolean_prompt_mask.repeat_interleave(num_samples_per_prompt, 0)
            
            
            
        #     # prompt_embeds = torch.cat([uncond_prompt_embeds, prompt_embeds])
        #     # boolean_prompt_mask = torch.cat([uncond_boolean_prompt_mask, boolean_prompt_mask])
            
        #     boolean_prompt_mask = (boolean_prompt_mask == 1).to(device)
        # else:
        encoded_text = self.encode_text(prompt)
        
        prompt_embeds = encoded_text['last_hidden_state']
        boolean_prompt_mask = encoded_text['attention_mask']
        subject_mask = encoded_text.get('subject_masks', None)
        
        if slider and subject_mask:
            prompt_embeds = slider.apply(prompt_embeds,subject_mask, scale = slider_scale)
        
        prompt_embeds = prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
        boolean_prompt_mask = boolean_prompt_mask.repeat_interleave(num_samples_per_prompt, 0)

        boolean_prompt_mask = (boolean_prompt_mask == 1).to(device)

        inference_scheduler.set_timesteps(num_steps, device=device)
        timesteps = inference_scheduler.timesteps

        num_channels_latents = self.unet_model_config["in_channels"]
        latents = self.prepare_latents(batch_size, inference_scheduler, num_channels_latents, prompt_embeds.dtype, device)

        num_warmup_steps = len(timesteps) - num_steps * inference_scheduler.order
        progress_bar = tqdm(range(num_steps), disable=disable_progress)
        
        if self.unet_model_config['infer_classifier_free_guidance_strength'] is not None:
            guidance_scale = self.unet_model_config['infer_classifier_free_guidance_strength']
        else:
            guidance_scale = guidance_scale

        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            # latent_model_input = torch.cat([latents] * 2) if classifier_free_guidance else latents
            latent_model_input = latents
            latent_model_input = inference_scheduler.scale_model_input(latent_model_input, t)

            # expand t to batch size
            bsz = latent_model_input.shape[0]
            time = torch.full((bsz,), t, dtype=torch.long, device=device)

            noise_pred = self.unet(
                latent_model_input, time = time, embedding=prompt_embeds, embedding_scale=guidance_scale, embedding_mask_proba=0
            )

            # perform guidance
            # if classifier_free_guidance:
            #     noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            #     noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = inference_scheduler.step(noise_pred, t, latents).prev_sample

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % inference_scheduler.order == 0):
                progress_bar.update(1)

        if self.set_from == "pre-trained":
            latents = self.group_out(latents.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
        return latents

    def prepare_latents(self, batch_size, inference_scheduler, num_channels_latents, dtype, device):
        shape = (batch_size, num_channels_latents, 64)
        latents = randn_tensor(shape, generator=None, device=device, dtype=dtype)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * inference_scheduler.init_noise_sigma
        return latents

    def encode_text_classifier_free(self, prompt, num_samples_per_prompt):
        device = next(self.parameters()).device
        
        
        # get tex embeddings for classifier free guidance
        encoded_text_dict = self.encode_text(prompt)
        prompt_embeds = encoded_text_dict['last_hidden_state']
        attention_mask = encoded_text_dict['attention_mask']
        subject_mask = encoded_text_dict.get('subject_masks', None)
                
        prompt_embeds = prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
        attention_mask = attention_mask.repeat_interleave(num_samples_per_prompt, 0)

        # get unconditional embeddings for classifier free guidance
        uncond_tokens = [""] * len(prompt)

        max_length = prompt_embeds.shape[1]
        uncond_encoded_text_dict = self.encode_text(uncond_tokens)
        negative_prompt_embeds = uncond_encoded_text_dict['last_hidden_state']
        uncond_attention_mask = uncond_encoded_text_dict['attention_mask']
                
        negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
        uncond_attention_mask = uncond_attention_mask.repeat_interleave(num_samples_per_prompt, 0)


        # For classifier free guidance, we need to do two forward passes.
        # We concatenate the unconditional and text embeddings into a single batch to avoid doing two forward passes
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        prompt_mask = torch.cat([uncond_attention_mask, attention_mask])
        boolean_prompt_mask = (prompt_mask == 1).to(device)
        
    

        return encoded_text_dict,uncond_encoded_text_dict
    
    
class LightningDiffGar(DiffGarLDM,LightningModule):
    
    def __init__(self,
                encoder_pair='clap',
                encoder_pair_kwargs=None,
                encoder_pair_ckpt=None,
                text_encoder = None,
                text_encoder_kwargs = None,
                text_encoder_ckpt = None,
                scheduler_name='stabilityai/stable-diffusion-2-1',
                scheduler_pred_type='epsilon',
                unet_model_config=None,
                unet_ckpt=None,
                snr_gamma=None,
                freeze_encoder_pair=True,
                freeze_unet=False,
                uncondition=False,
                subject_flagging = False,
                train_slider_concept = None,
                preextracted_latents = True,
                optimizer: OptimizerCallable = None,
                scheduler = None,
                
                ):#required for UNet architecture
        
        print("LightningDiffGar init")
        super().__init__(encoder_pair=encoder_pair,
                    encoder_pair_kwargs=encoder_pair_kwargs,
                    encoder_pair_ckpt=encoder_pair_ckpt,
                    scheduler_name=scheduler_name,
                    text_encoder=text_encoder,
                    text_encoder_kwargs=text_encoder_kwargs,
                    text_encoder_ckpt=text_encoder_ckpt,
                    scheduler_pred_type=scheduler_pred_type,
                    unet_model_config=unet_model_config,
                    unet_ckpt=unet_ckpt,
                    snr_gamma=snr_gamma,
                    freeze_encoder_pair=freeze_encoder_pair,
                    freeze_unet=freeze_unet,
                    uncondition=uncondition,
                    subject_flagging=subject_flagging,
                    train_slider_concept=train_slider_concept)
        
        self.preextracted_latents = preextracted_latents
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gen_examples = True
        self.first_run = True
        
    def training_step(self, batch, batch_idx):
        audio = batch['audio']
        prompt = batch['prompt']
        file_path  = batch['file_path']
        
        prompt_path = zip(prompt, file_path)
        # print prompt path tuples
        # for p, f in prompt_path:
        #     print(f'Prompt: {p}, File: {f}') if self.first_run else None
            
        
        
        if not self.preextracted_latents:
            latents = self.encoder_pair.get_audio_embedding_from_data(audio)
        else:
            latents = audio
            
        
        # text_embeds = self.encoder_pair.get_text_embedding(prompt)['projected_pooler_output']
        
        # # print(prompt)
        # # print(text_embeds)
        
        # sims = latents @ text_embeds.t()
        
        # sims = sims.max(dim=1).values
        # print(sims)
        
        # raise ValueError("Stopping after computing similarities")
        
        latents = latents.permute(0,2,1)
        loss = self(latents, prompt)
        
        
        
        
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        # log the learning rate
        self.log('learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=True)
        
        if self.global_step % 2000 == 0 and self.gen_examples:
            print(f"Generating some samples")
            preds = self.inference(prompt, self.inference_scheduler, num_steps = 50, disable_progress = False, guidance_scale = self.unet_model_config['infer_classifier_free_guidance_strength'])
            
            preds,latents = preds.permute(0,2,1), latents.permute(0,2,1)
            
            print(f'Generated samples of shape {preds.shape}') if self.first_run else None
            print(f'Ground truth samples of shape {latents.shape}') if self.first_run else None
            
            
            
            
            print(f"Computing CLAP score") if self.first_run else None
            gt_clap = self.encoder_pair.get_clap_score(latents, prompt, latents = True)['CLAP_Score']
            print(f"Ground truth CLAP score: {gt_clap}") if self.first_run else None
            print(f"Computing CLAP score for generated samples") if self.first_run else None
            pred_clap = self.encoder_pair.get_clap_score(preds, prompt, latents = True)['CLAP_Score']
            print(f"Generated CLAP score: {pred_clap}") if self.first_run else None
            self.log('gt_clap', gt_clap, on_step=True, on_epoch=True, prog_bar=True)
            self.log('pred_clap', pred_clap, on_step=True, on_epoch=True, prog_bar=True)
            
            # raise ValueError("Stopping after generating samples")
            
        if self.scheduler is not None:
            self.scheduler.step()
            
        return loss
        
    
    def validation_step(self, batch, batch_idx):
        audio = batch['audio']
        prompt = batch['prompt']
        
        if not self.preextracted_latents:
            latents = self.encoder_pair.get_audio_embedding_from_data(audio)
        else:
            latents = audio
        
        latents = latents.permute(0,2,1)
        loss = self(latents, prompt, validation_mode=True)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        if self.global_step % 10000 == 0:
            ## generate some samples and compute clap score
            pass
        
        return loss
    
    def configure_optimizers(self):
        if self.optimizer is None:
            optimizer = optim.Adam(
                self.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
        else:
            optimizer = self.optimizer(self.parameters())
            
        if self.scheduler is not None:
            # copy of the scheduler applied to the optimizer
            scheduler_class = eval(self.scheduler['class_name'])
            scheduler_kwargs = self.scheduler.get('init_args', {})
            scheduler = scheduler_class(optimizer, **scheduler_kwargs)
            self.scheduler = scheduler
            return [optimizer], [scheduler]
        
        return optimizer