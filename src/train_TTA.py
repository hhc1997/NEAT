import time
import logging
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
from ddp import *
import torch.distributed as dist
import torch.distributions as dists


def entropy_loss_against_noisy(outputs, eps=1e-3):
    entropys=softmax_entropy(outputs)
    entropy_threshold = torch.quantile(entropys, 0.1)
    weight=torch.clamp(torch.tensor(1.0) - entropys.clone().detach() /(entropy_threshold + eps), min=0) #0.1,2
    loss=entropys.mul(weight)[entropys<=entropy_threshold]
    return loss.mean(0)


@torch.no_grad()
def coarse_aligning_i2t(query_feat, gallery_feat_all, pos_gallery_feat_all, neg_gallery_feat_all):
    pos_scores_all = query_feat @ pos_gallery_feat_all.t()  # bs * all
    neg_scores_all = query_feat @ neg_gallery_feat_all.t()  # bs * all
    neg_relevance = torch.sigmoid(neg_scores_all)
    scores_all = pos_scores_all * (1 - neg_relevance)
    coarse_aligned_idxs = scores_all.argmax(dim=1)
    gallery_feat = gallery_feat_all[coarse_aligned_idxs]
    return gallery_feat


@torch.no_grad()
def coarse_aligning_t2i(query_feat, pos_query_feat, neg_query_feat, gallery_feat_all):
    scores_all_ = query_feat @ gallery_feat_all.t()
    pos_scores_all = pos_query_feat @ gallery_feat_all.t()  # bs * all
    neg_scores_all = neg_query_feat @ gallery_feat_all.t()  # bs * all
    neg_relevance = torch.relu(neg_scores_all)
    scores_all = pos_scores_all * (1 - neg_relevance)
    coarse_aligned_idxs = scores_all.argmax(dim=1)
    soft_t2i_labels = scores_all[:, coarse_aligned_idxs]
    gallery_feat = gallery_feat_all[coarse_aligned_idxs]

    return gallery_feat, coarse_aligned_idxs, soft_t2i_labels, scores_all_.argmax(dim=1), scores_all_


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)




def semantics_reversion_loss(scores, tau = 1):
    # compute image-sentence score matrix
    diagonal = scores.diag().view(scores.size(0), 1)
    d1 = diagonal.expand_as(scores)
    # compare every diagonal score to scores in its column
    cost_s = (- scores + d1).clamp(min=0)
    # cost_s = (margin - scores + d1)
    # clear diagonals
    mask = torch.eye(scores.size(0)) > 0.5
    mask = mask.to(cost_s.device)
    cost_s = cost_s.masked_fill_(mask, 0)
    cost_s_max = cost_s.max(1)[0]
    return cost_s_max.mean()/tau


def clip_loss(scores, logit_scale):
    logits_per_image = scores
    #logits_per_text = scores.T
    labels = torch.arange(logits_per_image.shape[0], device=scores.device, dtype=torch.long)
    total_loss = F.cross_entropy(logit_scale * logits_per_image, labels)

    return -total_loss


def textual_triplet_loss(text_feat, pos_text_feat, inv_text_feat):
    """
    Args:
        text_feat: 形状为(batch_size, D)的归一化文本特征， 含有否定约束，如 ''
        pos_text_feat: 形状为(batch_size, D)的归一化正向文本特征
        inv_text_feat: 形状为(batch_size, D)的归一化反向文本特征
    Returns:
        loss: 标量张量 - 计算得到的语义差异损失
    """

    pdist = nn.PairwiseDistance(p=2)
    loss_inv = torch.mean(2 - pdist(text_feat, inv_text_feat))  # \in [0,2]

    loss = torch.mean(pdist(text_feat, pos_text_feat))*5
    return loss + loss_inv






def train_one_epoch(model, tokenizer, data_loader_text, data_loader_image, optimizer, device, args):
    if args.distributed:
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    print("Computing features for evaluation...")

    len_image = len(data_loader_image.dataset.image)
    len_text = len(data_loader_text.dataset.text)

    score_matrix_i2t = torch.full((len_image, len_text), -100.0).to(device)
    score_matrix_t2i = torch.full((len_text, len_image), -100.0).to(device)

    model.eval()

    if args.tta_only_visual:
        pass
    else:
        if 'blip' in args.name:
            all_image_embeds = torch.zeros(len_image, 256).to(device)
        else:
            all_image_embeds = torch.zeros(len_image, 512).to(device)
        with torch.no_grad():
            for image, index in data_loader_image:
                image = image.to(device)
                image_embed = model.encode_image(image, normalize = True)
                all_image_embeds[index] = image_embed
        for i, (text, pos_text, neg_text, inv_text ,idx) in enumerate(tqdm(data_loader_text, desc="Processing", leave=False)):
            if 'blip' not in args.name:
                text = tokenizer(text).to(device)
                pos_text = tokenizer(pos_text).to(device)
                neg_text = tokenizer(neg_text).to(device)
                inv_text = tokenizer(inv_text).to(device)

            for step in range(args.tta_steps):
                text_feat = model.encode_text(text, normalize=True)
                pos_text_feat = model.encode_text(pos_text, normalize=True)
                neg_text_feat = model.encode_text(neg_text, normalize=True)
                inv_text_feat = model.encode_text(inv_text, normalize=True)
                coarse_img_feat, pred_t2i_idx, refined_t2i_labels, pred_t2i_idx_knn, sims_matrix =  coarse_aligning_t2i(text_feat, pos_text_feat, neg_text_feat, all_image_embeds)
                outputs = text_feat @ coarse_img_feat.t()
                outputs_aux = inv_text_feat @ coarse_img_feat.t()
                loss_entropy =  softmax_entropy(outputs.t()/ args.tta_temperature).mean(0)
                loss_sr = semantics_reversion_loss(outputs_aux, args.tta_temperature2)
                loss_tri = textual_triplet_loss(text_feat, pos_text_feat, inv_text_feat)
                loss =  loss_entropy  + loss_sr  + loss_tri

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            logging.info(f"\tloss_entropy: {loss_entropy.item():.3f}, loss_sr: {loss_sr.item():.3f}, loss_tri: {loss_tri.item():.3f}")
            # Evaluation
            score_matrix_t2i[idx] = sims_matrix


    return score_matrix_i2t, score_matrix_t2i

