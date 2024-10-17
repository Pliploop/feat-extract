from torch.utils.data import Dataset
from .loading_utils import *
import torch
import os
import random
from tqdm import tqdm
import pandas as pd



class TextAudioDataset(Dataset):
    def __init__(self,
                 annotations,
                 target_n_samples,
                 target_sr,
                 return_audio = True,
                 return_text = True,
                 concept = None,
                 return_full_audio = False,
                 preextracted_features = False,
                 truncate_preextracted = 50,
                 root_dir = None,
                 new_dir = None):
        self.annotations = annotations
        self.target_n_samples = target_n_samples
        self.target_sr = target_sr
        self.return_audio = return_audio
        self.return_text = return_text
        self.concept = concept
        self.return_full_audio = return_full_audio
        self.preextracted_features = preextracted_features
        self.truncate_preextracted = truncate_preextracted
        
        # get unique file name indices and caption indices for annotations
        annot_df = pd.DataFrame(annotations)
        # get label index for each unique file name
        
        ## get the index for each unique file name in order of appearance
        
        try:
            uniques = annot_df['file_path'].unique()
            annot_df['file_index'] = annot_df['file_path'].apply(lambda x: np.where(uniques == x)[0][0])
            
            
        except:
            pass
        
        for i, annot in enumerate(annotations):
            annot['file_index'] = annot_df.loc[i,'file_index']
            # annot['caption_index'] = annot_df.loc[i,'caption_index']
            
        if root_dir is not None and new_dir is not None:
            for annot in annotations:
                annot['file_path'] = annot['file_path'].replace(root_dir, new_dir)
        
        assert return_audio or return_text, "At least one of return_audio or return_text must be True (duh)"


    def purge(self):
        if self.return_audio and not self.preextracted_features:
            raise NotImplementedError("Purging your audio dataset is probably a bad idea")
        else:
            file_paths = [annot['file_path'] for annot in self.annotations]
            for file_path in file_paths:
                os.remove(file_path)
            print(f"Removed {len(file_paths)} files")
        
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx, return_full_audio = False, hop = None, verbose = False):
        
        
        return_full_audio = self.return_full_audio if return_full_audio is None else return_full_audio
        
        annot = self.annotations[idx]
        
        
        if self.return_audio:
            if not self.preextracted_features:
                audio = load_full_and_split(
                    annot['file_path'],
                    self.target_sr,
                    self.target_n_samples,
                    hop=hop,
                    verbose=verbose
                    ) if return_full_audio else load_audio_chunk(
                    annot['file_path'],
                    target_sr=self.target_sr,
                    target_n_samples=self.target_n_samples,
                    verbose=verbose
                    )
                audio = audio.mean(0,keepdim=True) if not return_full_audio else audio.mean(1,keepdim=True)
            else:
                try:
                    file_path = annot['file_path'].replace('.mp3','.npy').replace('.wav','.npy')
                    audio = np.load(file_path,mmap_mode='r')
                    if audio.shape[0] > self.truncate_preextracted:
                        rand_start = random.randint(0,audio.shape[0]-self.truncate_preextracted)
                        audio = audio[rand_start:rand_start+self.truncate_preextracted]
                        audio = torch.tensor(audio)
                    else:
                        return self[idx + 1]
                except Exception as e:
                    print(f"Error loading preextracted features: {e}") if verbose else None
                    return self[idx + 1]
        
        if self.return_text:
            possible_captions = annot['caption']
            # ramdomly choose a caption hash
            
            random_hash = random.choice(list(possible_captions.keys()))
        
            caption = possible_captions[random_hash]
            
            if self.concept is not None:
                
                    assert self.concept in annot['concepts'], f"Concept {self.concept} not found in annotations"
                
                    concept_captions = annot['concepts'][self.concept]
                    
                    #check if the hash is in the captions
                    if random_hash in concept_captions.keys():
                        concept_caption = concept_captions[random_hash]
                        pluscaption = random.choice(concept_caption['pluscaptions'])
                        minuscaption = random.choice(concept_caption['minuscaptions'])
                    else:
                        print(f"Hash {random_hash} of caption {caption} not found in concept {self.concept} captions")
                        return self.__getitem__(idx+1)
                    
        return_dict = {}
        
        if self.return_audio:
            return_dict['audio'] = audio
            return_dict['file_path'] = annot['file_path']
            
        if self.return_text:
            return_dict['prompt'] = caption
            if self.concept is not None:
                return_dict['plusprompt'] = pluscaption
                return_dict['minusprompt'] = minuscaption
                
        return_dict['file_idx'] = annot['file_index']

                
        return return_dict
    
    
    def extract_features(self, model, extract_method = 'extract_features', extract_kwargs = {}, out_key = 'embedding',hop = None, return_full_audio = True, verbose = False):
        
        device = next(model.parameters()).device
        print(f"Extracting features with {extract_method} method on {device} device") if verbose else None
        
        model.freeze()
        
        for i in range(len(self)):
            try:
                item = self.__getitem__(i, return_full_audio = return_full_audio, hop = hop, verbose = verbose)
                file_path = self.annotations[i]['file_path'].replace('.mp3','.npy').replace('.wav','.npy')
                
                audio = item['audio'].squeeze()
                
                if audio.shape[0] > 100:
                    chunks = torch.split(audio, 100, dim=0)
                    audio_features = []
                    for chunk in chunks:
                        original_device = chunk.device
                        chunk.to(device)
                        audio_features.append(getattr(model, extract_method)(chunk, **extract_kwargs)[out_key])
                        chunk.to(original_device)
                    audio_features = torch.cat(audio_features, dim=0)
                else:
                    audio_features = getattr(model, extract_method)(audio.to(device), **extract_kwargs)[out_key]
                
                
                
                print(f"Extracted features for {file_path}, shape: {audio_features.shape}") if verbose else None
                
                yield audio_features, file_path
            except Exception as e:
                print(f"Error extracting features for {file_path}: {e}") if verbose else None
                yield None, ''
        
    def extract_and_save_features(self, model, save_dir = None, extract_method = 'extract_features', extract_kwargs = {}, out_key = 'embedding', hop = None, return_full_audio = True, limit_n = None, save = False, verbose = True, root_path = None):
        
        
        print(self.__len__())
        
        audio_features_all = []
        counter = 0
        
        save_dir = '' if save_dir is None else save_dir
        
        if 's3://' in save_dir:
            import boto3
            import io
            client = boto3.client('s3')
        
        
        for audio_features, file_path in (pbar:= tqdm(self.extract_features(model, extract_method = extract_method, extract_kwargs = extract_kwargs, out_key = out_key, hop = hop, return_full_audio = return_full_audio, verbose = verbose))):
            
            # print(file_path, root_path, save_dir)
            
            if root_path is not None:
                file_path = file_path.replace(root_path+'/','')

            save_path = os.path.join(save_dir, file_path)
            
            if save and audio_features is not None:
                

                #remove the root path from the file path
                
                
                if 's3://' in save_dir:
                    bucket, key = save_dir.replace("s3://", "").split("/", 1)
                    key = f"{key}/{file_path}"
                    
                    # local_path = os.path.join(local_temp_dir, file_path)
                    # os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    # np.save(local_path, audio_features.detach().cpu().numpy())
                    
                    pbar.set_description(f"Uploading features to s3://{bucket}/{key}") if verbose else None
                    try:
                        # client.upload_file(save_path, bucket, key)
                        
                        buffer = io.BytesIO()
                        np.save(buffer, audio_features.detach().cpu().numpy())
                        buffer.seek(0)
                        client.put_object(Bucket=bucket, Key=key, Body=buffer)
                    except Exception as e:
                        print(f"Error uploading to s3: {e}") if verbose else None
                        
                    # os.remove(local_path)
                else:
                    pbar.set_description(f"Saving features in {save_path}, shape: {audio_features.shape}")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    if os.path.exists(save_path):
                        os.remove(save_path)
                    np.save(save_path, audio_features.detach().cpu().numpy())
                
                
                
            audio_features_all.append(audio_features.detach().cpu()) if audio_features is not None else None
            
            counter += 1
            if limit_n and counter >= limit_n:
                break
        try:   
            print(f"Returning {len(audio_features_all)} features") if verbose else None
            all_= torch.stack(audio_features_all)
            print(f"Stacked features, shape: {all_.shape}") if verbose else None
            return all_
        
        
        except:
            return None