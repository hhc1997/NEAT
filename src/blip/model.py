from .med import BertConfig, BertModel
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from torch import nn
import torch.nn.functional as F
from .blip import create_vit, init_tokenizer, load_checkpoint
from src.open_clip.transform import image_transform_v2
from src.open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from src.open_clip.utils import to_2tuple


class BLIP(nn.Module):
    def __init__(self,
                 med_config='/mnt/hanhc/negbench-main/benchmarks/src/blip/configs/med_config.json',
                 image_size=384,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 embed_dim=256,
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)
        text_width = self.text_encoder.config.hidden_size

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

    def encode_image(self, image, normalize: bool = False):
        image_embeds = self.visual_encoder(image)
        image_embeds = self.vision_proj(image_embeds[:, 0, :])
        image_feat = F.normalize(image_embeds, dim=1) if normalize else image_embeds
        return image_feat

    def encode_text(self, text, normalize: bool = False):
        text_input = self.tokenizer(text, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to('cuda')
        text_output = self.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask, mode='text')
        text_output = self.text_proj(text_output.last_hidden_state[:, 0, :])
        text_feat = F.normalize(text_output, dim=1) if normalize else text_output
        return text_feat



def creat_blip_model_and_transforms(pretrained='', **kwargs):
    model = BLIP(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
        print("missing keys:")
        print(msg.missing_keys)
    pp_cfg = PreprocessCfg()
    preprocess_train = image_transform_v2(pp_cfg, is_train=True)
    preprocess_val = image_transform_v2(pp_cfg, is_train=False)


    return model, preprocess_train, preprocess_val

@dataclass
class PreprocessCfg:
    size: Union[int, Tuple[int, int]] = 384
    mode: str = 'RGB'
    mean: Tuple[float, ...] = OPENAI_DATASET_MEAN
    std: Tuple[float, ...] = OPENAI_DATASET_STD
    interpolation: str = 'bicubic'
    resize_mode: str = 'shortest'
    fill_color: int = 0

    def __post_init__(self):
        assert self.mode in ('RGB',)

    @property
    def num_channels(self):
        return 3

    @property
    def input_size(self):
        return (self.num_channels,) + to_2tuple(self.size)

def freeze_blip_parameters(model, only_visual):
    model.train()
    model.requires_grad_(False)
    if only_visual:
        print("only_visual")
        for name, param in model.visual_encoder.named_parameters():
            if ('norm' in name) or ('Norm' in name):
                param.requires_grad_(True)
    else:
        print("only_text")
        for name, param in model.text_encoder.named_parameters():
            if ('norm' in name) or ('Norm' in name):
                param.requires_grad_(True)
    return model


def collect_blip_params(model, only_visual):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    if only_visual:
        for nm, m in model.visual_encoder.named_modules():
            if isinstance(m, (nn.LayerNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")
    else:
        for nm, m in model.text_encoder.named_modules():
            if isinstance(m, (nn.LayerNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")

    return params, names

