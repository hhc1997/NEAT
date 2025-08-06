import argparse
import ast


def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    model_name = model_name.lower()
    if "vit" in model_name:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        kw = {}
        for value in values:
            key, value = value.split('=')
            try:
                kw[key] = ast.literal_eval(value)
            except ValueError:
                kw[key] = str(value)  # fallback to string (avoid need to escape on command line)
        setattr(namespace, self.dest, kw)


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Path to file(s) with training data. When using webdataset, multiple datasources can be combined using the `::` separator.",
    )
    parser.add_argument(
        "--train-data-upsampling-factors",
        type=str,
        default=None,
        help=(
            "When using multiple data sources with webdataset and sampling with replacement, this can be used to upsample specific data sources. "
            "Similar to --train-data, this should be a string with as many numbers as there are data sources, separated by `::` (e.g. 1::2::0.5) "
            "By default, datapoints are sampled uniformly regardless of the dataset sizes."
        )
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to file(s) with validation data",
    )
    parser.add_argument(
        "--val-negated",
        default=False,
        action="store_true",
        help="Whether the validation data includes positive and negative objects."
    )
    parser.add_argument(
        "--train-negated",
        default=False,
        action="store_true",
        help="Whether the training batches explicitly include positive and negative images."
    )
    parser.add_argument(
        "--train-num-samples",
        type=int,
        default=None,
        help="Number of samples in dataset. Required for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--val-num-samples",
        type=int,
        default=None,
        help="Number of samples in dataset. Useful for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--dataset-type",
        choices=["webdataset", "csv", "synthetic", "auto"],
        default="auto",
        help="Which type of dataset to process."
    )
    parser.add_argument(
        "--dataset-resampled",
        default=False,
        action="store_true",
        help="Whether to use sampling with replacement for webdataset shard selection."
    )
    parser.add_argument(
        "--csv-separator",
        type=str,
        default="\t",
        help="For csv-like datasets, which separator to use."
    )
    parser.add_argument(
        "--csv-img-key",
        type=str,
        default="filepath",
        help="For csv-like datasets, the name of the key for the image paths."
    )
    parser.add_argument(
        "--csv-caption-key",
        type=str,
        default="title",
        help="For csv-like datasets, the name of the key for the captions."
    )
    parser.add_argument(
        "--imagenet-val",
        type=str,
        default=None,
        help="Path to imagenet val set for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--imagenet-v2",
        type=str,
        default=None,
        help="Path to imagenet v2 for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--synthetic-zeroshot",
        type=str,
        default=None,
        help="Path to synthetic dataset holdout set for conducting zero shot evaluation.",
    )
    parser.add_argument( # TODO: remove default value for the following 4 paths
        "--coco-zeroshot",
        type=str,
        default="/data/healthy-ml/scratch/kumail/projects/data/negation/COCO_val_multiclass.csv",
        help="Path to coco dataset holdout set for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--voc2007-zeroshot",
        type=str,
        default="/data/healthy-ml/scratch/kumail/projects/data/negation/VOC2007_multiclass.csv",
        help="Path to voc2007 dataset holdout set for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--coco-mcq",
        type=str,
        help="Path to coco dataset holdout set for conducting MCQ evaluation.",
    )
    parser.add_argument(
        "--voc2007-mcq",
        type=str,
        help="Path to voc2007 dataset holdout set for conducting MCQ evaluation.",
    )
    parser.add_argument(
        "--synthetic-mcq",
        type=str,
        help="Path to synthetic dataset holdout set for conducting MCQ evaluation.",
    )
    parser.add_argument(
        "--coco-retrieval",
        type=str,
        help="Path to coco dataset holdout set for conducting retrieval evaluation.",
    )
    parser.add_argument(
        "--coco-negated-retrieval",
        type=str,
        help="Path to coco dataset holdout set for conducting negation retrieval evaluation.",
    )
    parser.add_argument(
        "--coco-negated-tta",
        type=str,
        help="Path to pos, neg, and reversed coco dataset, used for TTA. ",
    )
    parser.add_argument(
        "--msrvtt-negated-tta",
        type=str,
        help="Path to pos, neg, and reversed msrvtt dataset, used for TTA. ",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        default=False,
        help="Use video data instead of images."
    )
    parser.add_argument(
        "--msrvtt-retrieval",
        type=str,
        help="Path to MSRVTT dataset holdout set for conducting video retrieval evaluation.",
    )
    parser.add_argument(
        "--msrvtt-negated-retrieval",
        type=str,
        help="Path to MSRVTT dataset holdout set for conducting negation video retrieval evaluation.",
    )
    parser.add_argument(
        "--msrvtt-mcq",
        type=str,
        help="Path to MSRVTT dataset holdout set for conducting MCQ evaluation.",
    )
    parser.add_argument(
        "--cxr-dataset",
        action="store_true",
        default=False,
        help="Use CXR dataset."
    )
    parser.add_argument(
        "--chexpert-mcq",
        type=str,
        default=None,
        help="Path to CheXpert dataset holdout set for conducting MCQ evaluation.",
    )
    parser.add_argument(
        "--chexpert-affirmation-mcq",
        type=str,
        default=None,
        help="Path to CheXpert dataset holdout set for conducting affirmation MCQ evaluation.",
    )
    # chexpert-binary-mcq
    parser.add_argument(
        "--chexpert-binary-mcq",
        type=str,
        # default="/data/healthy-ml/scratch/kumail/projects/data/negation/chexpert_binary_mcq.csv",
        default=None,
        help="Path to CheXpert dataset holdout set for conducting binary MCQ evaluation.",
    )
    # chexpert-affirmation-binary-mcq
    parser.add_argument(
        "--chexpert-affirmation-binary-mcq",
        type=str,
        # default="/data/healthy-ml/scratch/kumail/projects/data/negation/chexpert_binary_mcq_no_negation.csv",
        default=None,
        help="Path to CheXpert dataset holdout set for conducting affirmation binary MCQ evaluation.",
    )
    parser.add_argument(
        "--ham10000-mcq",
        type=str,
        default="/data/healthy-ml/scratch/kumail/projects/data/negation/ham10000_mcq.csv",
        help="Path to HAM10000 dataset holdout set for conducting MCQ evaluation.",
    )
    parser.add_argument(
        "--ham10000-affirmation-mcq",
        type=str,
        default="/data/healthy-ml/scratch/kumail/projects/data/negation/ham10000_mcq_no_negation.csv",
        help="Path to HAM10000 dataset holdout set for conducting affirmation MCQ evaluation.",
    )
    parser.add_argument(
        "--logs",
        type=str,
        default="./logs/",
        help="Where to store tensorboard logs. Use None to avoid storing logs.",
    )
    parser.add_argument(
        "--log-local",
        action="store_true",
        default=False,
        help="log files on local master, otherwise global master only.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Number of dataloader workers per GPU."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size per GPU."
    )
    parser.add_argument(
        "--epochs", type=int, default=32, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--epochs-cooldown", type=int, default=None,
        help="When scheduler w/ cooldown used, perform cooldown from total_epochs - cooldown_epochs onwards."
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=None, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")
    parser.add_argument(
        "--warmup", type=int, default=10000, help="Number of steps to warmup for."
    )
    parser.add_argument(
        "--use-bn-sync",
        default=False,
        action="store_true",
        help="Whether to use batch norm sync.")
    parser.add_argument(
        "--skip-scheduler",
        action="store_true",
        default=False,
        help="Use this flag to skip the learning rate decay.",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default='cosine',
        help="LR scheduler. One of: 'cosine', 'const' (constant), 'const-cooldown' (constant w/ cooldown). Default: cosine",
    )
    parser.add_argument(
        "--lr-cooldown-end", type=float, default=0.0,
        help="End learning rate for cooldown schedule. Default: 0"
    )
    parser.add_argument(
        "--lr-cooldown-power", type=float, default=1.0,
        help="Power for polynomial cooldown schedule. Default: 1.0 (linear decay)"
    )
    parser.add_argument(
        "--save-frequency", type=int, default=1, help="How often to save checkpoints."
    )
    parser.add_argument(
        "--save-most-recent",
        action="store_true",
        default=False,
        help="Always save the most recent model trained to epoch_latest.pt.",
    )
    parser.add_argument(
        "--zeroshot-frequency", type=int, default=2, help="How often to run zero shot."
    )
    parser.add_argument(
        "--val-frequency", type=int, default=1, help="How often to run evaluation with val data."
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "pure_bf16", "pure_fp16", "fp32"],
        default="amp",
        help="Floating point precision."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="RN50",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--pretrained",
        default='',
        type=str,
        help="Use a pretrained CLIP model weights with the specified tag or file path.",
    )
    parser.add_argument(
        "--pretrained-image",
        default=False,
        action='store_true',
        help="Load imagenet pretrained weights for image tower backbone if available.",
    )
    parser.add_argument(
        "--lock-image",
        default=False,
        action='store_true',
        help="Lock full image tower by disabling gradients.",
    )
    parser.add_argument(
        "--lock-image-unlocked-groups",
        type=int,
        default=0,
        help="Leave last n image tower layer groups unlocked.",
    )
    parser.add_argument(
        "--lock-image-freeze-bn-stats",
        default=False,
        action='store_true',
        help="Freeze BatchNorm running stats in image tower for any locked layers.",
    )
    parser.add_argument(
        '--image-mean', type=float, nargs='+', default=None, metavar='MEAN',
        help='Override default image mean value of dataset')
    parser.add_argument(
        '--image-std', type=float, nargs='+', default=None, metavar='STD',
        help='Override default image std deviation of of dataset')
    parser.add_argument(
        '--image-interpolation',
        default=None, type=str, choices=['bicubic', 'bilinear', 'random'],
        help="Override default image resize interpolation"
    )
    parser.add_argument(
        '--image-resize-mode',
        default=None, type=str, choices=['shortest', 'longest', 'squash'],
        help="Override default image resize (& crop) mode during inference"
    )
    parser.add_argument('--aug-cfg', nargs='*', default={}, action=ParseKwargs)
    parser.add_argument(
        "--grad-checkpointing",
        default=False,
        action='store_true',
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--local-loss",
        default=False,
        action="store_true",
        help="calculate loss w/ local features @ global (instead of realizing full global @ global matrix)"
    )
    parser.add_argument(
        "--gather-with-grad",
        default=False,
        action="store_true",
        help="enable full distributed gradient for feature gather"
    )
    parser.add_argument(
        '--force-image-size', type=int, nargs='+', default=None,
        help='Override default image size'
    )
    parser.add_argument(
        "--force-quick-gelu",
        default=False,
        action='store_true',
        help="Force use of QuickGELU activation for non-OpenAI transformer models.",
    )
    parser.add_argument(
        "--force-patch-dropout",
        default=None,
        type=float,
        help="Override the patch dropout during training, for fine tuning with no dropout near the end as in the paper",
    )
    parser.add_argument(
        "--force-custom-text",
        default=False,
        action='store_true',
        help="Force use of CustomTextCLIP model (separate text-tower).",
    )
    parser.add_argument(
        "--torchscript",
        default=False,
        action='store_true',
        help="torch.jit.script the model, also uses jit version of OpenAI models if pretrained=='openai'",
    )
    parser.add_argument(
        "--torchcompile",
        default=False,
        action='store_true',
        help="torch.compile() the model, requires pytorch 2.0 or later.",
    )
    parser.add_argument(
        "--trace",
        default=False,
        action='store_true',
        help="torch.jit.trace the model for inference / eval only",
    )
    parser.add_argument(
        "--accum-freq", type=int, default=1, help="Update the model every --acum-freq steps."
    )
    # arguments for distributed training
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--report-to",
        default='',
        type=str,
        help="Options are ['wandb', 'tensorboard', 'wandb,tensorboard']"
    )
    parser.add_argument(
        "--wandb-notes",
        default='',
        type=str,
        help="Notes if logging with wandb"
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default='open-clip',
        help="Name of the project if logging with wandb.",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged."
    )
    parser.add_argument(
        "--copy-codebase",
        default=False,
        action="store_true",
        help="If true, we copy the entire base on the log directory, and execute from there."
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training."
    )
    parser.add_argument(
        "--distributed",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--ddp-static-graph",
        default=False,
        action='store_true',
        help="Enable static graph optimization for DDP in PyTorch >= 1.11.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc)."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Default random seed."
    )
    parser.add_argument(
        "--grad-clip-norm", type=float, default=None, help="Gradient clip."
    )
    parser.add_argument(
        "--lock-text",
        default=False,
        action='store_true',
        help="Lock full text tower by disabling gradients.",
    )
    parser.add_argument(
        "--lock-text-unlocked-layers",
        type=int,
        default=0,
        help="Leave last n text tower layer groups unlocked.",
    )
    parser.add_argument(
        "--lock-text-freeze-layer-norm",
        default=False,
        action='store_true',
        help="Freeze LayerNorm running stats in text tower for any locked layers.",
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=100,
        help="Log every n steps to tensorboard/console/wandb.",
    )
    parser.add_argument(
        "--coca-caption-loss-weight",
        type=float,
        default=2.0,
        help="Weight assigned to caption loss in CoCa."
    )
    parser.add_argument(
        "--coca-contrastive-loss-weight",
        type=float,
        default=1.0,
        help="Weight assigned to contrastive loss when training CoCa."
    )
    parser.add_argument(
        "--remote-sync",
        type=str,
        default=None,
        help="Optinoally sync with a remote path specified by this arg",
    )
    parser.add_argument(
        "--remote-sync-frequency",
        type=int,
        default=300,
        help="How frequently to sync to a remote directly if --remote-sync is not None.",
    )
    parser.add_argument(
        "--remote-sync-protocol",
        choices=["s3", "fsspec"],
        default="s3",
        help="How to do the remote sync backup if --remote-sync is not None.",
    )
    parser.add_argument(
        "--delete-previous-checkpoint",
        default=False,
        action="store_true",
        help="If true, delete previous checkpoint after storing a new one."
    )
    parser.add_argument(
        "--distill-model",
        default=None,
        help='Which model arch to distill from, if any.'
    )
    parser.add_argument(
        "--distill-pretrained",
        default=None,
        help='Which pre-trained weights to distill from, if any.'
    )
    parser.add_argument(
        "--use-bnb-linear",
        default=None,
        help='Replace the network linear layers from the bitsandbytes library. '
        'Allows int8 training/inference, etc.'
    )
    parser.add_argument(
        "--siglip",
        default=False,
        action="store_true",
        help='Use SigLip (sigmoid) loss.'
    )
    # Add argument for mcq_train_data
    parser.add_argument(
        "--mcq-train-data",
        type=str,
        default=None,
        help="Path to file(s) with training data for the MCQ task."
    )
    parser.add_argument(
        "--contrastive-loss-weight",
        type=float,
        default=1.0,
        help="Weight for the contrastive loss term."
    )
    parser.add_argument(
        "--mcq-loss-weight",
        type=float,
        default=1.0,
        help="Weight for the MCQ loss term."
    )
    parser.add_argument(
        "--train-separate-negated-data",
        type=str,
        default=None,
        help="Path to file(s) with synthetic negated training data. It should be a csv dataset.",
    )

    parser.add_argument(
        "--negated-dataset-type",
        choices=["webdataset", "csv", "synthetic", "explicit", "auto"],
        default="csv",
        help="Which type of dataset to process."
    )

    parser.add_argument(
        "--negation-alpha",
        type=float,
        default=1.0,
        help="Weight for contrastive loss term in synthetic data objective."
    )

    parser.add_argument(
        "--negation-beta",
        type=float,
        default=1.0,
        help="Weight for dot product term in synthetic data objective."
    )

    parser.add_argument(
        "--natural-weight",
        type=float,
        default=1.0,
        help="Weight for natural batch loss term."
    )

    parser.add_argument(
        "--synthetic-weight",
        type=float,
        default=1.0,
        help="Weight for synthetic batch loss term."
    )

    ## new for TTA method
    parser.add_argument(
        "--tta-retrieval",
        type=str,
        default='t2i'
    )

    parser.add_argument(
        "--tta-data-type",
        type=str,
        choices=['coco_negated', 'msrvtt_negated'],
        default='coco_negated'
    )

    parser.add_argument(
        "--tta-init-lr",
        type=float,
        default=2e-4
    )

    parser.add_argument(
        "--tta-wd",
        type=float,
        default=1e-3
    )

    parser.add_argument(
        "--tta-backbone",
        type=str,
        default='clip'
    )

    parser.add_argument(
        "--tta-total-bs",
        type=int,
        default=64
    )

    parser.add_argument(
        "--tta-steps",
        type=int,
        default=3
    )

    parser.add_argument(
        "--tta-temperature",
        type=float,
        default=0.03
    )

    parser.add_argument(
        "--tta-temperature2",
        type=float,
        default=0.07
    )



    args = parser.parse_args(args)

    # If some params are not passed, we use the default values based on model name.
    default_params = get_default_params(args.model)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)

    return args