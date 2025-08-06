import ast
import json
import logging
import math
import os
import random
import sys
import braceexpand
from dataclasses import dataclass
from multiprocessing import Value

import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample

from .video_utils.video_dataset import CsvVideoCaptionDataset, CsvVideoMCQDataset

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from torch.utils.data import default_collate

def image_captions_collate_fn(batch):
    '''
    Custom collate function for image captioning tasks
    '''
    batch_items = list(zip(*batch))
    images = default_collate(batch_items[0])
    return (images, *batch_items[1:])



class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t", tokenizer=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenize([str(self.captions[idx])])[0]
        return images, texts

class CsvCLassDataset(Dataset):
    def __init__(self, csv_file, transforms, sep=',', img_key='positive_filepath', target_key='target'):
        """
        Dataset for classification tasks (useful for zero-shot and linear probing evals)

        Args:
            csv_file (string): Path to the csv file with annotations.
            transforms (callable): Transform to be applied on a sample.
            sep (string): Separator used in the csv file.
            img_key (string): Column name for image file paths.
            target_key (string): Column name for target labels.
        """
        self.df = pd.read_csv(csv_file, sep=sep)
        self.transforms = transforms
        self.images = self.df[img_key].tolist()
        self.labels = self.df[target_key].tolist()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images = Image.open(str(self.images[idx]))
        images = self.transforms(images)
        labels = self.labels[idx]
        return images, labels

class CsvMCQDataset(Dataset):
    def __init__(self, csv_file, transforms, num_answers=4, path="image_path", tokenizer=None):
        """
        Dataset for MCQ task evaluation

        Args:
            csv_file (string): Path to the csv file with annotations.
            transforms (callable): Transform to be applied on a sample.
            num_answers (int): Number of answer choices (captions) given to the model.
            path (string): Column name for image file paths.
            tokenizer (callable): Tokenizer to be applied on the captions.
                Needs to be passed if dataset is used for training.
        """
        self.df = pd.read_csv(csv_file, sep=',')
        self.transforms = transforms
        self.num_answers = num_answers
        self.path = path
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row[self.path]
        captions = [row[f"caption_{i}"] for i in range(self.num_answers)]
        correct_answer = row["correct_answer"]
        correct_answer_template = row["correct_answer_template"]

        images = self.transforms(Image.open(image_path))

        if self.tokenizer is not None:
            captions = [self.tokenizer([str(caption)])[0] for caption in captions]
            captions = torch.stack(captions) # (num_answers, max_seq_len)

        return images, captions, correct_answer, correct_answer_template

class CsvBinaryMCQDataset(Dataset):
    def __init__(self, csv_file, transforms):
        """
        Dataset for Binary MCQ task evaluation (2 options: caption_0 and caption_1)

        Args:
            csv_file (string): Path to the csv file with annotations.
            transforms (callable): Transform to be applied on a sample.
        """
        self.df = pd.read_csv(csv_file, sep=',')
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["image_path"]
        captions = [row["caption_0"], row["caption_1"]]
        correct_answer = row["correct_answer"]

        # Load and transform the image
        image = self.transforms(Image.open(image_path))
        return image, captions, correct_answer

class CsvImageCaptionDataset(Dataset):
    def __init__(self, csv_file, transforms, sep=',', img_key='filepath', caption_key='captions'):
        """
        Dataset for image captioning or retrieval tasks

        Args:
            csv_file (string): Path to the csv file with image paths and captions.
            transforms (callable): Transform to be applied on a sample.
            sep (string): Separator used in the csv file.
            img_key (string): Column name for image file paths.
            caption_key (string): Column name for captions.
        """
        self.df = pd.read_csv(csv_file, sep=sep)
        self.transforms = transforms
        self.img_key = img_key
        self.caption_key = caption_key

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df.iloc[idx][self.img_key]
        images = self.transforms(Image.open(image_path))
        captions = eval(self.df.iloc[idx][self.caption_key])
        return images, captions


class CsvImageCaptionDataset_Sep(Dataset):
    def __init__(self, csv_file, transforms, sep=',', img_key='filepath', caption_key_pos='pos_captions', caption_key_neg='neg_captions'):
        """
        Dataset for image captioning or retrieval tasks

        Args:
            csv_file (string): Path to the csv file with image paths and captions.
            transforms (callable): Transform to be applied on a sample.
            sep (string): Separator used in the csv file.
            img_key (string): Column name for image file paths.
            caption_key (string): Column name for captions.
        """
        self.df = pd.read_csv(csv_file, sep=sep)

        # self.df[caption_key_pos] = self.df[caption_key_pos].apply(ast.literal_eval)
        # self.df[caption_key_neg] = self.df[caption_key_neg].apply(ast.literal_eval)
        self.transforms = transforms
        self.img_key = img_key
        self.caption_key_pos = caption_key_pos
        self.caption_key_neg = caption_key_neg

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df.iloc[idx][self.img_key]
        images = self.transforms(Image.open(image_path))
        captions_pos = eval(self.df.iloc[idx][self.caption_key_pos])
        captions_neg = eval(self.df.iloc[idx][self.caption_key_neg])
        return images, captions_pos, captions_neg


class CsvValset_Negation(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t", tokenizer=None, test_prompts_filename="/data/healthy-ml/scratch/kumail/projects/data/coco/annotations/coco_rule_based_test_prompts.csv"):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        logging.debug(f'Loading csv data from {test_prompts_filename}.')
        test_captions_df = pd.read_csv(test_prompts_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.test_prompts = test_captions_df["prompt"].tolist()
        self.transforms = transforms

        self.positive_objects = df["positive_objects"].tolist()
        self.negative_objects = df["negative_objects"].tolist()
        self.test_positive_objects = test_captions_df["positive_objects"].tolist()
        self.test_negative_objects = test_captions_df["negative_objects"].tolist()


        # Initialize a set to hold unique strings
        unique_strings = set()

        # Function to parse and update unique strings set
        def parse_and_update_unique_strings(string_list):
            for string_repr in string_list:
                # Convert the string representation of a list to an actual list
                actual_list = ast.literal_eval(string_repr)
                # Update the set of unique strings with the elements from this list
                unique_strings.update(actual_list)

        # Parse and update unique strings from both lists
        parse_and_update_unique_strings(self.positive_objects)
        parse_and_update_unique_strings(self.negative_objects)

        # Convert the set back to a list if you need a list structure
        unique_labels = list(unique_strings)

        label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
        # Now, you can map the labels in self.positive_objects and self.negative_objects to integers
        self.positive_objects = [[label_to_int[label] for label in ast.literal_eval(obj_list)] for obj_list in self.positive_objects]
        self.negative_objects = [[label_to_int[label] for label in ast.literal_eval(obj_list)] for obj_list in self.negative_objects]

        logging.debug('Done loading data.')

        self.tokenize = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenize([str(self.captions[idx])])[0]
        positive_objects = str(self.positive_objects[idx])
        # negative_objects = str(self.negative_objects[idx])

        return images, texts, positive_objects
    
    def get_test_prompts(self):
        texts = self.tokenize(self.test_prompts)
        return texts, self.test_positive_objects, self.test_negative_objects # TODO: check what the method actually returns

class CsvTrainset_Negation(CsvValset_Negation):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t", tokenizer=None):
        # Call the initializer of the parent class (CsvValset_Negation)
        super().__init__(input_filename, transforms, img_key, caption_key, sep, tokenizer)

    def __getitem__(self, idx):
        image = self.transforms(Image.open(str(self.images[idx])))
        text = self.tokenize([str(self.captions[idx])])[0]
        negative_objects = self.negative_objects[idx]

        # Select one label from negative_objects at random
        selected_label = random.choice(negative_objects)

        # Search for an index in self.positive_objects that contains the same label
        idx_ = next(i for i, pos_objs in enumerate(self.positive_objects) if selected_label in pos_objs)

        # Load the second image and tokenize its corresponding text
        image_ = self.transforms(Image.open(str(self.images[idx_])))
        text_ = self.tokenize([str(self.captions[idx_])])[0]

        # Return the pairs of images and texts
        return (image, text), (image_, text_)

class CsvTrainset_Explicit_Negation(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t", tokenizer=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        # Load positive AND negative data
        self.positive_images = df[f"positive_{img_key}"].tolist()
        self.negative_images = df[f"negative_{img_key}"].tolist()
        self.positive_captions = df[f"positive_{caption_key}"].tolist()
        self.negative_captions = df[f"negative_{caption_key}"].tolist()
        self.transforms = transforms
        
        logging.debug('Done loading data.')

        self.tokenize = tokenizer

    def __len__(self):
        return len(self.positive_captions)

    def __getitem__(self, idx):
        image = self.transforms(Image.open(str(self.positive_images[idx])))
        text = self.tokenize([str(self.positive_captions[idx])])[0]

        # Load the second image and tokenize its corresponding text
        image_ = self.transforms(Image.open(str(self.negative_images[idx])))
        text_ = self.tokenize([str(self.negative_captions[idx])])[0]

        # Return the pairs of images and texts
        return (image, text), (image_, text_)
        
class InfiniteDataLoader:
    """This class allows to iterate the dataloader infinitely batch by batch.
    When there are no more batches the iterator is reset silently.
    This class allows to keep the memory of the state of the iterator hence its
    name.
    """

    def __init__(self, dataloader):
        """This initialization takes a dataloader and creates an iterator object
        from it.

        Parameters
        ----------
        dataloader : torch.utils.data.dataloader
            A dataloader object built from one of the datasets of this repository.
        """
        self._dataloader = dataloader

        self._iterator = iter(self._dataloader)

    def _reset_iterator(self):
        self._iterator = iter(self._dataloader)

    def __len__(self):
        return len(self._dataloader.dataset)

    def get_samples(self):
        """This method generates the next batch from the iterator or resets it
        if needed. It can be called an infinite amount of times.

        Returns
        -------
        tuple
            a batch from the iterator, including the input and output
        """
        try:
            batch = next(self._iterator)
        except StopIteration:
            self._reset_iterator()
            batch = next(self._iterator)
        return batch

class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split('::')
        assert len(weights) == len(urllist),\
            f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


def get_dataset_size(shards):
    shards_list, _ = expand_urls(shards)
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    else:
        if is_train:
            data_path = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        weights=None,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert len(self.urls) == len(self.weights),\
                f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(url=self.rng.choices(self.urls, weights=self.weights, k=1)[0])


def get_wds_dataset(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None):
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_shards = None
    if is_train:
        if args.train_num_samples is not None:
            num_samples = args.train_num_samples
        else:
            num_samples, num_shards = get_dataset_size(input_shards)
            if not num_samples:
                raise RuntimeError(
                    'Currently, the number of dataset samples must be specified for the training dataset. '
                    'Please specify it via `--train-num-samples` if no dataset length info is present.')
    else:
        # Eval will just exhaust the iterator if the size is not specified.
        num_samples = args.val_num_samples or 0 

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc

    if is_train and args.train_data_upsampling_factors is not None:
        assert resampled, "--train_data_upsampling_factors is only supported when sampling with replacement (with --dataset-resampled)."
    
    if resampled:
        pipeline = [ResampledShards2(
            input_shards,
            weights=args.train_data_upsampling_factors,
            deterministic=True,
            epoch=shared_epoch,
        )]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    pipeline.extend([
        wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png;jpeg;webp", text="txt"),
        wds.map_dict(image=preprocess_img, text=lambda text: tokenizer(text)[0]),
        wds.to_tuple("image", "text"),
        wds.batched(args.batch_size, partial=not is_train)
    ])

    dataset = wds.DataPipeline(*pipeline)

    if is_train:
        if not resampled:
            num_shards = num_shards or len(expand_urls(input_shards)[0])
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)

def get_separate_negated_dataset(args, input_filename, preprocess_fn, is_train, epoch=0, tokenizer=None):
    if is_train:
        if args.negated_dataset_type == "explicit": # TODO test this function
            dataset = CsvTrainset_Explicit_Negation(
                input_filename,
                preprocess_fn,
                img_key=args.csv_img_key,
                caption_key=args.csv_caption_key,
                sep=args.csv_separator,
                tokenizer=tokenizer
            )
        else:
            dataset = CsvTrainset_Negation(
                input_filename,
                preprocess_fn,
                img_key=args.csv_img_key,
                caption_key=args.csv_caption_key,
                sep=args.csv_separator,
                tokenizer=tokenizer
            )

    else:
        dataset = CsvValset_Negation(
            input_filename,
            preprocess_fn,
            img_key=args.csv_img_key,
            caption_key=args.csv_caption_key,
            sep=args.csv_separator,
            tokenizer=tokenizer
        )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size // 2,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    if is_train:
        infinitedataloader = InfiniteDataLoader(dataloader)

    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(infinitedataloader, sampler)


def get_csv_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    if args.train_negated and is_train:
        dataset = CsvTrainset_Negation(
            input_filename,
            preprocess_fn,
            img_key=args.csv_img_key,
            caption_key=args.csv_caption_key,
            sep=args.csv_separator,
            tokenizer=tokenizer
        )

    elif args.val_negated and not is_train:
        dataset = CsvValset_Negation(
            input_filename,
            preprocess_fn,
            img_key=args.csv_img_key,
            caption_key=args.csv_caption_key,
            sep=args.csv_separator,
            tokenizer=tokenizer
        )
    else:
        dataset = CsvDataset(
            input_filename,
            preprocess_fn,
            img_key=args.csv_img_key,
            caption_key=args.csv_caption_key,
            sep=args.csv_separator,
            tokenizer=tokenizer
        )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    if args.train_negated and is_train:
        dataloader = InfiniteDataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.workers,
            pin_memory=True,
            sampler=sampler
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.workers,
            pin_memory=True,
            sampler=sampler,
            drop_last=is_train,
        )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def get_csv_mcq_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    if not is_train:
        # Not implemented 
        raise NotImplementedError("get_csv_mcq_dataset is only supported for training.")
    input_filename = args.mcq_train_data
    assert input_filename, "MCQ data filename must be provided."

    dataset = CsvMCQDataset(
        csv_file=input_filename,
        transforms=preprocess_fn,
        tokenizer=tokenizer
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def get_eval_dataset(args, input_filename, preprocess_fn, mode, img_key='positive_filepath', target_key='target'):
    if mode == 'classification':
        dataset = CsvCLassDataset(csv_file=input_filename, transforms=preprocess_fn, img_key=img_key, target_key=target_key)
    elif mode == 'retrieval':
        if args.video:
            # TODO: update the csv file to have a more intuitive column name
            dataset = CsvVideoCaptionDataset(csv_file=input_filename, transforms=preprocess_fn, video_key='filepath', caption_key='captions')
        else:
            dataset = CsvImageCaptionDataset(csv_file=input_filename, transforms=preprocess_fn)
    elif mode == 'separated_retrieval':
        if args.video:
            # TODO:
            pass
        else:
            dataset = CsvImageCaptionDataset_Sep(csv_file=input_filename, transforms=preprocess_fn, caption_key_pos='pos_captions', caption_key_neg='neg_captions')

    elif mode == 'mcq':
        if args.video:
            # TODO: update the csv file to have a more intuitive column name
            dataset = CsvVideoMCQDataset(csv_file=input_filename, transforms=preprocess_fn, video_key='image_path')
        else:     
            dataset = CsvMCQDataset(csv_file=input_filename, transforms=preprocess_fn)
    elif mode == 'binary_mcq':
        dataset = CsvBinaryMCQDataset(csv_file=input_filename, transforms=preprocess_fn)
    else:
        raise ValueError(f"Unsupported mode: {mode} for evaluation dataset.")

    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None

    collate_fn = image_captions_collate_fn if mode == 'retrieval' or mode == 'separated_retrieval' else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        collate_fn=collate_fn
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


class SyntheticDataset(Dataset):

    def __init__(
            self,
            transform=None,
            image_size=(224, 224),
            caption="Dummy caption",
            dataset_size=100,
            tokenizer=None,
    ):
        self.transform = transform
        self.image_size = image_size
        self.caption = caption
        self.image = Image.new('RGB', image_size)
        self.dataset_size = dataset_size

        self.preprocess_txt = lambda text: tokenizer(text)[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.transform is not None:
            image = self.transform(self.image)
        return image, self.preprocess_txt(self.caption)


def get_synthetic_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    image_size = preprocess_fn.transforms[0].size
    dataset = SyntheticDataset(
        transform=preprocess_fn, image_size=image_size, dataset_size=args.train_num_samples, tokenizer=tokenizer)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "csv_mcq":
        return get_csv_mcq_dataset  # New function to handle MCQ CSV datasets
    elif dataset_type == "synthetic":
        return get_synthetic_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        elif ext in ['tar']:
            return get_wds_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extension {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    

def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.video:
        if args.msrvtt_retrieval:
            data["msrvtt-retrieval"] = get_eval_dataset(args, args.msrvtt_retrieval, preprocess_val, 'retrieval')

        if args.msrvtt_negated_retrieval:
            data["msrvtt-negated-retrieval"] = get_eval_dataset(args, args.msrvtt_negated_retrieval, preprocess_val, 'retrieval')

        if args.msrvtt_mcq:
            data["msrvtt-mcq"] = get_eval_dataset(args, args.msrvtt_mcq, preprocess_val, 'mcq')

        return data

    if args.cxr_dataset: # TODO: rename to something more intuitive, like args.medical_dataset
        if args.chexpert_mcq:
            data["chexpert-mcq"] = get_eval_dataset(args, args.chexpert_mcq, preprocess_val, 'mcq')

        if args.chexpert_affirmation_mcq:
            data["chexpert-affirmation-mcq"] = get_eval_dataset(args, args.chexpert_affirmation_mcq, preprocess_val, 'mcq')

        if args.chexpert_affirmation_binary_mcq:
            data["chexpert-binary-mcq"] = get_eval_dataset(args, args.chexpert_binary_mcq, preprocess_val, 'binary_mcq') # TODO
            data["chexpert-affirmation-binary-mcq"] = get_eval_dataset(args, args.chexpert_affirmation_binary_mcq, preprocess_val, 'binary_mcq')

        return data

    if args.train_data or args.dataset_type == "synthetic":
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)
        
    # Check if MCQ training data is provided
    if args.mcq_train_data:
        data["mcq_train"] = get_dataset_fn(args.mcq_train_data, 'csv_mcq')(
            args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)
        
    if args.train_separate_negated_data:
        data["train_negated"] = get_separate_negated_dataset(
            args, args.train_separate_negated_data, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)


    if args.synthetic_mcq:
        data["synthetic-mcq"] = get_eval_dataset(args, args.synthetic_mcq, preprocess_val, 'mcq', img_key='filepath', target_key='targets')
    
    if args.coco_mcq:
        data["coco-mcq"] = get_eval_dataset(args, args.coco_mcq, preprocess_val, 'mcq', img_key='filepath', target_key='targets')

    if args.coco_retrieval:
        data["coco-retrieval"] = get_eval_dataset(args, args.coco_retrieval, preprocess_val, 'retrieval')

    if args.coco_negated_retrieval:
        data["coco-negated-retrieval"] = get_eval_dataset(args, args.coco_negated_retrieval, preprocess_val, 'retrieval')


    if args.voc2007_mcq:
        data["voc2007-mcq"] = get_eval_dataset(args, args.voc2007_mcq, preprocess_val, 'mcq', img_key='filepath', target_key='targets')

    return data
