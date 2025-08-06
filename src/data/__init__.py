"""
preparing dataset for [TTA], not the data code in the evaluation dir.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from data.coco_negated_dataset import coco_negated_retrieval_eval_image,  coco_negated_retrieval_eval_text
from data.msrvtt_negated_dataset import msrvtt_negated_retrieval_eval_image, msrvtt_negated_retrieval_eval_text

def create_dataset(dataset_name, input_filename, resize=384):
    
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    transform_test = transforms.Compose([
        transforms.Resize((resize, resize),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])

    transform_test_video =  transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
        transforms.Resize((resize, resize), interpolation=InterpolationMode.BICUBIC, antialias=True),
        normalize,
    ])

    if dataset_name=='coco_negated':

        test_text_dataset = coco_negated_retrieval_eval_text(transform_test, csv_file=input_filename, sep=',',
                                                             img_key='filepath', caption_key_pos='pos_captions',
                                                             caption_key_neg='neg_captions')
        test_image_dataset = coco_negated_retrieval_eval_image(transform_test, csv_file=input_filename, sep=',',
                                                               img_key='filepath', caption_key_pos='pos_captions',
                                                               caption_key_neg='neg_captions')
        return test_text_dataset, test_image_dataset  

    elif dataset_name=='msrvtt_negated':
        test_text_dataset = msrvtt_negated_retrieval_eval_text(transform_test_video, csv_file=input_filename, sep=',',
                                                             img_key='filepath', caption_key_pos='pos_captions',
                                                             caption_key_neg='neg_captions')
        test_image_dataset = msrvtt_negated_retrieval_eval_image(transform_test_video, csv_file=input_filename, sep=',',
                                                                 img_key='filepath', caption_key_pos='pos_captions',
                                                                 caption_key_neg='neg_captions')
        return test_text_dataset, test_image_dataset



def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers     


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
        shuffle = True
        drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    
