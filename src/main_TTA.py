import logging
import os
import json
os.environ["HF_ENDPOINT"]= 'https://hf-mirror.com'
from transformers import CLIPProcessor, CLIPModel
import re
import sys
import random
import ruamel.yaml
yaml = ruamel.yaml.YAML(typ='rt')
import numpy as np
import torch

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

from src.open_clip import create_model_and_transforms, get_tokenizer, create_model_from_pretrained
from src.open_clip import freeze_parameters, collect_params
from src.blip.model import creat_blip_model_and_transforms, freeze_blip_parameters, collect_blip_params
from src.blip.blip import init_tokenizer
from src.utils.data import get_data
from src.utils.distributed import is_master, init_distributed_device
from src.utils.logger import setup_logging

from params import parse_args
from train_TTA import train_one_epoch


from src.evaluation.utils import evaluate, evaluate_video
from data import create_dataset, create_sampler, create_loader

import torch.multiprocessing
import tta_utils

torch.multiprocessing.set_sharing_strategy('file_descriptor')

LATEST_CHECKPOINT_NAME = "epoch_latest.pt"


def random_seed(seed=0, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def main(args):
    args = parse_args(args)

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # fully initialize distributed device environment
    device = init_distributed_device(args)

    if args.name is None:
        # Terminate session with error message
        raise ValueError("Please provide a name for the evaluation run")
        return -1

    # Set up logging
    log_base_path = os.path.join(args.logs, args.name)
    args.log_path = None
    if is_master(args, local=args.log_local):
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)

    # Setup text logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # Setup wandb, tensorboard, checkpoint logging
    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    if is_master(args):
        args.tensorboard_path = os.path.join(log_base_path, "tensorboard") if args.tensorboard else ''
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''

    # Load model
    if isinstance(args.force_image_size, (tuple, list)) and len(args.force_image_size) == 1:
        # arg is nargs, single (square) image size list -> int
        args.force_image_size = args.force_image_size[0]
    random_seed(args.seed, 0)
    model_kwargs = {}
    if args.siglip:
        model_kwargs['init_logit_scale'] = np.log(10)  # different from CLIP
        model_kwargs['init_logit_bias'] = -10

    if args.cxr_dataset and "plip" in args.name:
        print("Loading model from pretrained: BioMedCLIP-PubMedBERT_256-vit_base_patch16_224")
        model = CLIPModel.from_pretrained("vinid/plip")
        model_NEATed, preprocess_train, preprocess_val = create_model_and_transforms(
            args.model,
            args.pretrained,
            precision=args.precision,
            device=device,
            jit=args.torchscript,
            force_quick_gelu=args.force_quick_gelu,
            force_custom_text=args.force_custom_text,
            force_patch_dropout=args.force_patch_dropout,
            force_image_size=args.force_image_size,
            image_mean=args.image_mean,
            image_std=args.image_std,
            image_interpolation=args.image_interpolation,
            image_resize_mode=args.image_resize_mode,  # only effective for inference
            aug_cfg=args.aug_cfg,
            pretrained_image=args.pretrained_image,
            output_dict=True,
            video=args.video,  # TODO: add video support in parser
            **model_kwargs,
        )

    elif "blip" in args.name:
        print("Loading model from pretrained: blip")
        model, preprocess_train, preprocess_val = creat_blip_model_and_transforms(pretrained=args.pretrained, image_size=384,
                                vit='base', vit_grad_ckpt=True, vit_ckpt_layer=4)
        model = model.to(device)
    else:
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            args.model,
            args.pretrained,
            precision=args.precision,
            device=device,
            jit=args.torchscript,
            force_quick_gelu=args.force_quick_gelu,
            force_custom_text=args.force_custom_text,
            force_patch_dropout=args.force_patch_dropout,
            force_image_size=args.force_image_size,
            image_mean=args.image_mean,
            image_std=args.image_std,
            image_interpolation=args.image_interpolation,
            image_resize_mode=args.image_resize_mode,  # only effective for inference
            aug_cfg=args.aug_cfg,
            pretrained_image=args.pretrained_image,
            output_dict=True,
            video=args.video,  # TODO: add video support in parser
            **model_kwargs,
        )
    random_seed(args.seed, args.rank)
    if is_master(args):
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    # initialize datasets
    start_epoch = 0

    if "blip" in args.name:
        tokenizer = init_tokenizer()
    else:
        tokenizer = get_tokenizer(args.model)
    data = get_data(
        args,
        (preprocess_train, preprocess_val),
        epoch=start_epoch,
        tokenizer=tokenizer,
    )
    assert len(data), 'At least one train or eval dataset must be specified.'
    print("data keys:")
    for key in data.keys():
        print(key)

    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        if args.val_data is not None:
            args.val_sz = data["val"].dataloader.num_samples
        # you will have to configure this for your project!
        wandb.init(
            project=args.wandb_project_name,
            name=args.name,
            # id=args.name,
            notes=args.wandb_notes,
            tags=[],
            resume='auto' if args.resume == "latest" else None,
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log='all')
        wandb.save(params_file)
        logging.debug('Finished loading wandb.')

    # Pytorch 2.0 adds '_orig_mod.' prefix to keys of state_dict() of compiled models.
    # For compatibility, we save state_dict() of the original model, which shares the
    # weights without the prefix.
    original_model = model
    if args.torchcompile:
        logging.info('Compiling model...')
        model = torch.compile(original_model)

    if 'train' not in data:
        # If using int8, convert to inference mode.
        if args.use_bnb_linear is not None:
            from open_clip.utils import convert_int8_model_to_inference_mode
            convert_int8_model_to_inference_mode(model)
        #Evaluate. If you want to evaluate models without TTA
        # if args.video:
        #     evaluate_video(model, data, start_epoch, args, tb_writer=writer, tokenizer=tokenizer)
        # else:
        #     evaluate(model, data, start_epoch, args, tb_writer=writer, tokenizer=tokenizer)
        # return

    # Start our TTA method
    if args.tta_retrieval is not None:

        print("Creating retrieval dataset") # select coco_negated or msrvtt_negated
        #tta_data = create_dataset('msrvtt_negated', args.msrvtt_negated_tta, resize=224)
        tta_data = create_dataset('coco_negated', args.coco_negated_tta, resize=384 if 'blip' in args.name else 224)


        tta_data_text, tta_data_image = tta_data if isinstance(tta_data, tuple) else (None, None)
        if args.distributed:
            nt, gr = tta_utils.get_world_size(), tta_utils.get_rank()
            if args.retrieval == 't2i':
                samplers = create_sampler([tta_data_text], [True], nt, gr) + create_sampler([tta_data_image], [False], nt, gr)
            elif args.retrieval == 'i2t':
                samplers = create_sampler([tta_data_text], [False], nt, gr) + create_sampler([tta_data_image], [True], nt, gr)
            else:
                samplers = [None, None]
            bs_text = bs_image = args.tta_total_bs // nt
        else:
            samplers = [None, None]
            bs_text = bs_image = args.tta_total_bs
        loader_text, loader_image = create_loader([tta_data_text, tta_data_image], samplers,
                                                  batch_size=[bs_text, bs_image],
                                                  num_workers=[1, 1], is_trains=[False, False],
                                                  collate_fns=[None, None])

        print("Creating model")
        args.tta_only_visual = True if args.tta_retrieval == 'i2t' else False
        if 'blip' in args.name:
            model = freeze_blip_parameters(model, only_visual=args.tta_only_visual)
            trainable_params = collect_blip_params(model, only_visual=args.tta_only_visual)[0]
        else:
            model = freeze_parameters(model, only_visual=args.tta_only_visual)
            trainable_params = collect_params(model, only_visual=args.tta_only_visual)[0]

        optimizer =  torch.optim.AdamW(params=trainable_params, lr=args.tta_init_lr, weight_decay=args.tta_wd)

        print("Start Test Time Adaptation")
        score_matrix_i2t, score_matrix_t2i = train_one_epoch(model, tokenizer, loader_text, loader_image, optimizer, device, args)

        print('Eval after TTA')

    #Evaluate.
    if args.video:
        evaluate_video(model, data, start_epoch, args, score_matrix_t2i = score_matrix_t2i, tb_writer=writer, tokenizer=tokenizer)
    else:
        evaluate(model, data, start_epoch, args, score_matrix_t2i = score_matrix_t2i, tb_writer=writer, tokenizer=tokenizer)


if __name__ == "__main__":
    m = sys.argv
    main(sys.argv[1:])