from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from hashlib import sha256
import pandas as pd
from .datasets import TextAudioDataset
import os
from sklearn.model_selection import train_test_split


def get_folder_annotations(data_path = None):
    
    # recursively get all files in the data_path directory that are audio files, and their paths
    audio_files = []
    
    for root, dirs, files in os.walk(data_path):
        audio_files += [os.path.join(root, file) for file in files if file.endswith('.wav') or file.endswith('.mp3')]
        
    records = [{'file_path': file, 'caption': '', 'split': 'train'} for file in audio_files]
    
    return records
    
    

class AudioDataModule(LightningDataModule):
    
    def __init__(self, task, task_kwargs = {}, return_audio = True, return_text = True, concept = None, target_n_samples = 96000, target_sr = 48000, batch_size = 32, num_workers = 0, preextracted_features = False, truncate_preextracted = 50, root_dir = None, new_dir = None):


        super().__init__()

        self.annotations = eval(f"get_{task}_annotations")(**task_kwargs) if task is not None else get_folder_annotations(**task_kwargs)


        self.return_audio = return_audio
        self.return_text = return_text
        self.concept = concept
        self.target_n_samples = target_n_samples
        self.target_sr = target_sr
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.preextracted_features = preextracted_features
        self.truncate_preextracted = truncate_preextracted
        
        self.root_dir = root_dir
        self.new_dir = new_dir # for when the dataset was extracted to a new directory


        # do some cleaning : we want to return a list of dictionary records.
        # each record has a file_path as a string. captions are stored as a dictionary of possible captions
        # with keys being hashes of the captions and values being the captions themselves.
        # let's start by dealing with the case where captions are strings instead, let's turn them into lists of strings

        self.train_annotations = [annot for annot in self.annotations if annot['split'] == 'train']
        self.val_annotations = [annot for annot in self.annotations if annot['split'] == 'val']
        self.test_annotations = [annot for annot in self.annotations if annot['split'] == 'test']
        ##  
        
    def setup(self, stage: str) -> None:
        self.train_dataset = TextAudioDataset(annotations=self.train_annotations, target_n_samples=self.target_n_samples, target_sr=self.target_sr, return_audio=self.return_audio, return_text=self.return_text, concept=self.concept, preextracted_features=self.preextracted_features, truncate_preextracted=self.truncate_preextracted, root_dir=self.root_dir, new_dir=self.new_dir)
        self.val_dataset = TextAudioDataset(annotations=self.val_annotations, target_n_samples=self.target_n_samples, target_sr=self.target_sr, return_audio=self.return_audio, return_text=self.return_text, concept=self.concept, preextracted_features=self.preextracted_features, truncate_preextracted=self.truncate_preextracted, root_dir=self.root_dir, new_dir=self.new_dir)
        self.test_dataset = TextAudioDataset(annotations=self.test_annotations, target_n_samples=self.target_n_samples, target_sr=self.target_sr, return_audio=self.return_audio, return_text=self.return_text, concept=self.concept, preextracted_features=self.preextracted_features, truncate_preextracted=self.truncate_preextracted, root_dir=self.root_dir, new_dir=self.new_dir)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)