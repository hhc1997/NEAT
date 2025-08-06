import os
import ast
import pandas as pd
import json
import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from PIL import Image
from .utils import pre_caption

class coco_negated_retrieval_eval_image(Dataset):
    def __init__(self, transform, csv_file, sep=',', img_key='filepath', caption_key = 'captions', caption_key_pos='pos_captions', caption_key_neg='neg_captions', max_words=77):
        '''
        image_root (string): Root directory of images (e.g. flickr30k/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        self.df = pd.read_csv(csv_file, sep=sep)
        self.df['text'] = self.df[caption_key].apply(self.safe_eval)
        self.df['image'] = self.df[img_key]

        self.annotation = self.df[['image', 'text']].to_dict('records')

        self.transform = transform


        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['text']):
                self.text.append(pre_caption(caption, max_words))
                self.img2txt[img_id].append(txt_id)
                # self.txt2img[txt_id] = img_id
                self.txt2img[txt_id] = []
                self.txt2img[txt_id].append(img_id)
                txt_id += 1

    def safe_eval(self, x):
        if isinstance(x, str):
            return ast.literal_eval(x)
        return x

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        image_path = self.annotation[index]['image']
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, index


class coco_negated_retrieval_eval_text(Dataset):
    def __init__(self, transform, csv_file, sep=',', img_key='filepath', caption_key = 'captions', caption_key_pos='pos_captions', caption_key_neg='neg_captions',
                 caption_key_inv = 'inverted_captions', caption_key_ori = 'ori_captions',max_words=77):
        '''
        image_root (string): Root directory of images (e.g. flickr30k/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        self.df = pd.read_csv(csv_file, sep=sep)


        self.df['text'] = self.df[caption_key].apply(self.safe_eval)
        self.df['pos_text'] = self.df[caption_key_pos].apply(self.safe_eval)
        self.df['neg_text'] = self.df[caption_key_neg].apply(self.safe_eval)
        self.df['inv_text'] = self.df[caption_key_inv].apply(self.safe_eval)
        self.df['image'] = self.df[img_key]

        self.annotation = self.df[['image', 'text', 'pos_text', 'neg_text', 'inv_text']].to_dict('records')
        self.transform = transform


        self.text = []
        self.pos_text = []
        self.neg_text = []
        # self.ori_text = []
        self.inv_text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            captions = ann['text']
            pos_captions = ann['pos_text']
            neg_captions = ann['neg_text']
            inv_captions = ann['inv_text']
            # Limit to 5 captions per image
            for i, caption in enumerate(captions):
                self.text.append(pre_caption(caption, max_words))
                self.pos_text.append(pre_caption(pos_captions[i], max_words))
                self.neg_text.append(pre_caption(neg_captions[i], max_words))
                self.inv_text.append(pre_caption(inv_captions[i], max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = []
                self.txt2img[txt_id].append(img_id)
                txt_id += 1

    def safe_eval(self, x):
        if isinstance(x, str):
            return ast.literal_eval(x)
        return x

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):

        caption = self.text[index]
        pos_caption = self.pos_text[index]
        neg_caption = self.neg_text[index]
        inv_caption = self.inv_text[index]
        return caption, pos_caption, neg_caption, inv_caption, index
