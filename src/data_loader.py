import os
from typing import List, Union
from collections import OrderedDict

import numpy as np
import pandas as pd

from PIL import Image
import spacy

import torch
from torch.nn.utils.rnn import pad_sequence 
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

"""
TODO:
- smarter batching (short, medium, long see Tutorial 3)

"""

class Vocabulary():
    """
    Numericalize vocabulary
    
    .fit() to create the word -> index mapping
    
    .apply() to numericalize a text
    
    """
    
    
    def __init__(self, 
                 unk_cutoff=5,
                 unk_label='<UNK>',
                 start_label='<START>',
                 end_label='<END>',
                 pad_label='<PAD>',
                 tokenizer='en_core_web_sm',
                ):

        self.unk_cutoff = unk_cutoff
        
        self.unk_label = unk_label
        self.start_label = start_label
        self.end_label = end_label
        self.pad_label = pad_label

        self.idx_to_word = dict()
        self.word_to_idx = dict()
        self.__initialize()
        
        self.tokenizer = spacy.load(tokenizer)
        
        # Misc
        self.word_frequencies = OrderedDict()
        #todo: number of unk

    def __initialize(self):
        self.word_to_idx[self.pad_label] = 0
        self.word_to_idx[self.start_label] = 1
        self.word_to_idx[self.end_label] = 2
        self.word_to_idx[self.unk_label] = 3
        self.idx_to_word = {v:k for k,v in self.word_to_idx.items()}
        
    def fit(self, sentences:List[str]):
        """
        sentences = ['i love dogs', 'cats are ok too']
        """
        sentences = [self.tokenize_sentence(s) for s in sentences]
        
        # Drop infrequent words
        words, counts = np.unique(flatten(sentences), return_counts=True)
        word_counts = pd.Series(counts, index=words)
        self.word_frequencies = word_counts.sort_values(ascending=False).to_dict()
        word_counts = word_counts[word_counts >= self.unk_cutoff]
        
        #Update word-index dicts
        max_idx = max(self.idx_to_word.keys())
        self.word_to_idx.update({w:i+max_idx+1 for i,w in enumerate(word_counts.index)})
        self.idx_to_word = {v:k for k,v in self.word_to_idx.items()}
        
    def apply(self, sentences:Union[List[str], str], add_start_end=True):
        """
        todo: remove?
        sentences either list of sentences or one sentence.
        """
        if isinstance(sentences, str):
            return self.numericalize_sentence(sentences, add_start_end)
        else:
            return [self.numericalize_sentence(s, add_start_end) for s in sentences]

        
    def numericalize_sentence(self, sentence:str, add_start_end=True):
        tokens = self.tokenize_sentence(sentence)
        num_sentence = [self.word_to_idx.get(t, self.word_to_idx[self.unk_label]) for t in tokens]
        if add_start_end:
            num_sentence.insert(0, self.word_to_idx[self.start_label])
            num_sentence.append(self.word_to_idx[self.end_label])
        return num_sentence
        
    def tokenize_sentence(self, sentence:str):
        # "i don't like dogs" -> [i, do, n't, like, dogs]
        sentence = sentence.lower()
        tokens = [t.text for t in list(self.tokenizer.tokenizer(sentence))]
        return tokens
    
    def get_word_frequencies(self, apply_unk_cutoff=True):
        if  apply_unk_cutoff:
            return OrderedDict({k:v for k,v in self.word_frequencies.items() if v>=self.unk_cutoff})
        else:
            return self.word_frequencies
        


class FlickrDataset(Dataset):
    #https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    # Dataset needs: __init__, __len__, __getitem__
    
    def __init__(self,
                 images_dir,
                 captions_path,
                 transform=None,
                 unk_cutoff=5,
                 unk_label='<UNK>',
                 start_label='<START>',
                 end_label='<END>',
                 pad_label='<PAD>'
                ):
        
        self.images_dir = images_dir
        self.captions_path = captions_path
        
        image_captions = pd.read_csv(self.captions_path, delimiter = ',') #image, caption
        self.images = image_captions['image'] #only image ids
        self.captions = image_captions['caption']
        
        self.unk_cutoff = unk_cutoff
        self.unk_label = unk_label
        self.start_label = start_label
        self.end_label = end_label
        self.pad_label = pad_label
        
        self.vocab = Vocabulary(
            unk_cutoff=self.unk_cutoff,
            unk_label= self.unk_label,
            start_label=self.start_label,
            end_label=self.end_label,
            pad_label=self.pad_label
        )
        
        self.vocab.fit(self.captions.to_list())
        
        self.transform = transform
        
    def __getitem__(self, idx):
        caption = self.captions.iloc[idx]
        image_id = self.images.iloc[idx]
        img = Image.open(os.path.join(self.images_dir, image_id)).convert("RGB")

        if self.transform is not None: #todo #same size etc?
            img = self.transform(img)

        numericalized_caption = self.vocab.apply(caption, add_start_end=True)

        return img, torch.tensor(numericalized_caption)
    
    def __len__(self):
        return len(self.captions)

    
class Collate():
    """
    Collate to apply the padding to the captions with dataloader
    """
    def __init__(self, pad_idx, batch_first=False):
        self.pad_idx = pad_idx
        self.batch_first = batch_first
    def __call__(self, batch):
        #.unsqueeze(0) to make concatable by correct dimension.
        # Assume equal size images, otherwise torch.cat does not work.
        images = torch.cat([i[0].unsqueeze(0) for i in batch], dim=0)
        # batch_first=False each caption is a column
        captions = pad_sequence([i[1] for i in batch], batch_first=self.batch_first, padding_value=self.pad_idx)
        return images, captions
    
def get_loader(
                dataset,
                transform=None,
                batch_size=32,
                num_workers=4,
                batch_first=False,
                shuffle=True,
                pin_memory=True,
            ):
    
    """
    batch_first: specifies which dimension batches should span
    shuffle: shuffle data
    
    """
    
    pad_idx = dataset.vocab.word_to_idx[dataset.vocab.pad_label]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=Collate(pad_idx=pad_idx, batch_first=batch_first)
    )
    
    return loader

    
# library

def flatten(l):
    rt = []
    for i in l:
        if isinstance(i,list): rt.extend(flatten(i))
        else: rt.append(i)
    return rt

if __name__ == '__main__':

    img_dir = '../resources/flickr8k/Images'
    captions = '../resources/flickr8k/captions.txt'
    loader, dataset = get_loader(
        img_dir, captions
    )

    l = len(loader)
    for idx, (imgs, captions) in enumerate(loader):
        print(f'{idx}/{l}', imgs.shape)
        print(f'{idx}/{l}', captions.shape)
