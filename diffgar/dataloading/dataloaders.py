from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from hashlib import sha256
import pandas as pd
from .datasets import TextAudioDataset
import os
from sklearn.model_selection import train_test_split


def get_song_describer_annotations(data_path = None, csv_path = None, val_split = 0.1):
    
    data_path = data_path if data_path is not None else '/import/research_c4dm/jpmg86/song-describer/data/audio'
    csv_path = os.path.join(os.path.dirname(data_path), 'song_describer.csv') if csv_path is None else csv_path
    
    df = pd.read_csv(csv_path)
    
    
    df = df[['path','caption','is_valid_subset']].rename(columns = {'path':'file_path'})
    df['file_path'] = os.path.join(data_path) + '/' + df['file_path']
    #replace .mp3 with .2min.mp3
    df['file_path'] = df['file_path'].apply(lambda x: x.replace('.mp3','.2min.mp3'))
    
    records = df.to_dict(orient = 'records')
    
    for record in records:
        record['caption'] = {sha256(record['caption'].encode('utf-8')).hexdigest(): record['caption']}
    
    
    if val_split == 0.0:
        print('No validation split')
        for record in records:
            record['split'] = 'train'
        return records
    
    train_indices, val_indices = train_test_split(range(len(records)), test_size = val_split, random_state = 42)
    
    for idx in train_indices:
        records[idx]['split'] = 'train'
        
    for idx in val_indices:
        records[idx]['split'] = 'val'
    
    return records



def get_musiccaps_annotations(data_path = None, csv_path = None, val_split = 0.1, test_split = 0.1):
        
        df = pd.read_csv(csv_path)
        df['file_path'] = data_path + '/' + df['ytid'] + '.wav'
        
        records = df.to_dict(orient = 'records')
        
        for record in records:
            record['caption'] = {sha256(record['caption'].encode('utf-8')).hexdigest(): record['caption']}
            
        if val_split == 0.0:
            print('No validation split')
            for record in records:
                record['split'] = 'train'
            return records
        
        # split into train, val, test
        train_indices, test_indices = train_test_split(range(len(records)), test_size = test_split + val_split, random_state = 42)
        val_indices, test_indices = train_test_split(test_indices, test_size = test_split/(test_split + val_split), random_state = 42)
        
        
        
        for idx in train_indices:
            records[idx]['split'] = 'train'
            
        for idx in val_indices:
            records[idx]['split'] = 'val'
            
        for idx in test_indices:
            records[idx]['split'] = 'test'
            
        return records
            

def get_upmm_annotations(data_path = None, csv_path = None):
    
    data_path = data_path if data_path is not None else '/import/research_c4dm/jpmg86/upmm/data/audio'
    csv_path = os.path.join(os.path.dirname(data_path), 'upmm_captions.csv') if csv_path is None else csv_path
    
    df = pd.read_csv(csv_path)
    
    df['file_path'] = os.path.join(data_path) + '/' + df['file_path']
    
    records = df.to_dict(orient = 'records')
    
    for record in records:
        record['caption'] = {sha256(record['caption'].encode('utf-8')).hexdigest(): record['caption']}
    
    return records

def get_folder_annotations(data_path = None):
    
    # recursively get all files in the data_path directory that are audio files, and their paths
    
    for root, dirs, files in os.walk(data_path):
        audio_files = [os.path.join(root, file) for file in files if file.endswith('.wav') or file.endswith('.mp3')]
        
    records = [{'file_path': file, 'caption': '', 'split': 'train'} for file in audio_files]
    
    return records
    
    

class TextAudioDataModule(LightningDataModule):
    
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

        for annot in self.annotations:
            if isinstance(annot['caption'], str):
                annot['caption'] = [annot['caption']]

        for annot in self.annotations:
            if isinstance(annot['caption'], list):
                annot['caption'] = {sha256(caption.encode('utf-8')).hexdigest(): caption for caption in annot['caption']}   
        
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