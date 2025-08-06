import logging

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.open_clip import get_input_dtype, get_tokenizer
from src.utils.precision import get_autocast


def evaluate_model(model, dataloader, args, tokenizer, recall_k_list=[1, 5, 10]):
    """
    Evaluates a model on retrieval tasks.
    Args:
        model: torch.nn.Module
            The model to evaluate.
        dataloader: torch.utils.data.DataLoader
            The dataloader to evaluate on.
        args: argparse.Namespace
            The command line arguments.
        tokenizer: transformers.PreTrainedTokenizer
            The tokenizer to use.
        recall_k_list: list
            A list of k values for recall@k.
    Returns:
        metrics: dict
            The evaluation metrics.
    """
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)
    if tokenizer is None:
        tokenizer = get_tokenizer(args.model)
    model.eval()

    batch_images_emb_list = []
    batch_texts_emb_list = []
    texts_image_index = []
    dataloader = dataloader_with_indices(dataloader)

    # Iterate over the dataloader
    for batch_images, batch_texts, inds in tqdm(dataloader):
        batch_images = batch_images.to(device=args.device, dtype=input_dtype)
        if 'blip' in args.name:
            batch_texts_tok = None
        else:
            batch_texts_tok = tokenizer([text for i, texts in enumerate(batch_texts) for text in texts]).to(args.device)
        batch_texts_image_index = [ind for ind, texts in zip(inds, batch_texts) for text in texts]

        # Compute the embeddings
        with torch.no_grad(), autocast():
            batch_images_emb = F.normalize(model.encode_image(batch_images), dim=-1)
            if 'blip' in args.name:
                flat_batch_texts = []
                for texts in batch_texts:
                    flat_batch_texts.extend(texts)
                batch_texts_emb = model.encode_text(flat_batch_texts, normalize = True)
            else:
                batch_texts_emb = F.normalize(model.encode_text(batch_texts_tok), dim=-1) # 320 * 512

        # Append the embeddings and indices
        batch_images_emb_list.append(batch_images_emb.cpu())
        batch_texts_emb_list.append(batch_texts_emb.cpu())
        texts_image_index.extend(batch_texts_image_index)

    batch_size = len(batch_images_emb_list[0])
    images_emb = torch.cat(batch_images_emb_list)
    texts_emb = torch.cat(batch_texts_emb_list)

    # Compute the scores
    scores = texts_emb @ images_emb.t()
    positive_pairs = torch.zeros_like(scores, dtype=bool)
    positive_pairs[torch.arange(len(scores)), texts_image_index] = True
    metrics = {}

    # Compute the recall@k
    for recall_k in recall_k_list:
        metrics[f"image_retrieval_recall@{recall_k}"] = (
                    batchify(recall_at_k, scores, positive_pairs, batch_size, args.device,
                             k=recall_k) > 0).float().mean().item()
        metrics[f"text_retrieval_recall@{recall_k}"] = (
                    batchify(recall_at_k, scores.T, positive_pairs.T, batch_size, args.device,
                             k=recall_k) > 0).float().mean().item()

    # Compute median rank (medR)
    # 直接计算
    metrics["image_retrieval_medR"] = median_rank(scores, positive_pairs)
    metrics["text_retrieval_medR"] = median_rank(scores.T, positive_pairs.T)

    return metrics

def retrieval_eval(model, data, args, scores, tokenizer=None):
    """
    Evaluates a model on retrieval tasks.
    Args:
        model: torch.nn.Module
            The model to evaluate.
        data: dict
            A dictionary of datasets.
        args: argparse.Namespace
            The command line arguments.
        tokenizer: transformers.PreTrainedTokenizer
            The tokenizer to use.
    Returns:
        results: dict
            The evaluation results.
    """
    results = {}

    if args.video:
        if 'msrvtt-retrieval' in data:
            logging.info('Evaluating on the MSR-VTT retrieval task')
            msrvtt_retrieval = evaluate_model(model, data['msrvtt-retrieval'].dataloader, args, tokenizer=tokenizer, recall_k_list=[1, 5, 10])
            results['msrvtt-t2i-R@1'] = msrvtt_retrieval['image_retrieval_recall@1']
            results['msrvtt-t2i-R@5'] = msrvtt_retrieval['image_retrieval_recall@5']
            results['msrvtt-t2i-R@10'] = msrvtt_retrieval['image_retrieval_recall@10']

            results['msrvtt-i2t_R@1'] = msrvtt_retrieval['text_retrieval_recall@1']
            results['msrvtt-i2t-R@5'] = msrvtt_retrieval['text_retrieval_recall@5']
            results['msrvtt-i2t-R@10'] = msrvtt_retrieval['text_retrieval_recall@10']
        
        if 'msrvtt-negated-retrieval' in data:
            logging.info('Evaluating on the MSR-VTT negated retrieval task')
            msrvtt_negated_retrieval_online = evaluate_model_with_similarity(data['msrvtt-negated-retrieval'].dataloader, args, scores, recall_k_list=[1, 5, 10])
            results['online-msrvtt-negated-t2i-R@1'] = msrvtt_negated_retrieval_online['image_retrieval_recall@1']
            results['online-msrvtt-negated-t2i-R@5'] = msrvtt_negated_retrieval_online['image_retrieval_recall@5']
            results['online-msrvtt-negated-t2i-R@10'] = msrvtt_negated_retrieval_online['image_retrieval_recall@10']

            results['online-msrvtt-negated-i2t-R@1'] = msrvtt_negated_retrieval_online['text_retrieval_recall@1']
            results['online-msrvtt-negated-i2t-R@5'] = msrvtt_negated_retrieval_online['text_retrieval_recall@5']
            results['online-msrvtt-negated-i2t-R@10'] = msrvtt_negated_retrieval_online['text_retrieval_recall@10']


            msrvtt_negated_retrieval_offline = evaluate_model(model, data['msrvtt-negated-retrieval'].dataloader, args, tokenizer=tokenizer, recall_k_list=[1, 5, 10])
            results['offline-msrvtt-negated-t2i-R@1'] = msrvtt_negated_retrieval_offline['image_retrieval_recall@1']
            results['offline-msrvtt-negated-t2i-R@5'] = msrvtt_negated_retrieval_offline['image_retrieval_recall@5']
            results['offline-msrvtt-negated-t2i-R@10'] = msrvtt_negated_retrieval_offline['image_retrieval_recall@10']

            results['offline-msrvtt-negated-i2t-R@1'] = msrvtt_negated_retrieval_offline['text_retrieval_recall@1']
            results['offline-msrvtt-negated-i2t-R@5'] = msrvtt_negated_retrieval_offline['text_retrieval_recall@5']
            results['offline-msrvtt-negated-i2t-R@10'] = msrvtt_negated_retrieval_offline['text_retrieval_recall@10']

    else:
        if 'coco-retrieval' in data:
            logging.info('Evaluating on the COCO retrieval task')
            coco_retrieval = evaluate_model(model, data['coco-retrieval'].dataloader, args, tokenizer=tokenizer, recall_k_list=[1, 5, 10])
            results['coco-t2i-R@1'] = coco_retrieval['image_retrieval_recall@1']
            results['coco-t2i-R@5'] = coco_retrieval['image_retrieval_recall@5']
            results['coco-t2i-R@10'] = coco_retrieval['image_retrieval_recall@10']
            # results['coco-t2i_medR'] = coco_retrieval["image_retrieval_medR"]

            results['coco-i2t-R@1'] = coco_retrieval['text_retrieval_recall@1']
            results['coco-i2t-R@5'] = coco_retrieval['text_retrieval_recall@5']
            results['coco-i2t-R@10'] = coco_retrieval['text_retrieval_recall@10']
            # results['coco-i2t_medR'] = coco_retrieval["text_retrieval_medR"]

        if 'coco-negated-retrieval' in data:
            logging.info('Evaluating on the COCO negated retrieval task')
            coco_negated_retrieval_online = evaluate_model_with_similarity(data['coco-negated-retrieval'].dataloader, args, scores, recall_k_list=[1, 5, 10])
            results['online-coco-negated-t2i-R@1'] = coco_negated_retrieval_online['image_retrieval_recall@1']
            results['online-coco-negated-t2i-R@5'] = coco_negated_retrieval_online['image_retrieval_recall@5']
            results['online-coco-negated-t2i-R@10'] = coco_negated_retrieval_online['image_retrieval_recall@10']
            #results['online-coco-negated_t2i_medR'] = coco_negated_retrieval_online["image_retrieval_medR"]

            results['online-coco-negated-i2t-R@1'] = coco_negated_retrieval_online['text_retrieval_recall@1']
            results['online-coco-negated-i2t-R@5'] = coco_negated_retrieval_online['text_retrieval_recall@5']
            results['online-coco-negated-i2t-R@10'] = coco_negated_retrieval_online['text_retrieval_recall@10']
            #results['online-coco-negated_i2t_medR'] = coco_negated_retrieval_online["text_retrieval_medR"]

            coco_negated_retrieval_offline = evaluate_model(model, data['coco-negated-retrieval'].dataloader, args, tokenizer=tokenizer, recall_k_list=[1, 5, 10])
            results['offline-coco-negated-t2i-R@1'] = coco_negated_retrieval_offline['image_retrieval_recall@1']
            results['offline-coco-negated-t2i-R@5'] = coco_negated_retrieval_offline['image_retrieval_recall@5']
            results['offline-coco-negated-t2i-R@10'] = coco_negated_retrieval_offline['image_retrieval_recall@10']
            # results['offline-coco-negated_t2i_medR'] = coco_negated_retrieval_offline["image_retrieval_medR"]

            results['offline-coco-negated-i2t-R@1'] = coco_negated_retrieval_offline['text_retrieval_recall@1']
            results['offline-coco-negated-i2t-R@5'] = coco_negated_retrieval_offline['text_retrieval_recall@5']
            results['offline-coco-negated-i2t-R@10'] = coco_negated_retrieval_offline['text_retrieval_recall@10']
            # results['offline-coco-negated_i2t_medR'] = coco_negated_retrieval_offline["text_retrieval_medR"]




    return results


def median_rank(scores, positive_pairs):
    """
    计算median rank指标

    Args:
        scores: 相似度矩阵 (n_queries, n_targets)
        positive_pairs: 布尔矩阵，标记正样本对 (n_queries, n_targets)

    Returns:
        median_rank: 中位数排名
    """
    # 对分数进行降序排序，获取排序后的索引
    sorted_indices = torch.argsort(scores, dim=1, descending=True)

    # 创建排名矩阵
    ranks = torch.zeros_like(sorted_indices)
    for i in range(len(scores)):
        ranks[i, sorted_indices[i]] = torch.arange(len(scores[0]), device=scores.device)

    # 获取正样本的排名（注意：排名从0开始，所以需要+1）
    positive_ranks = (ranks + 1) * positive_pairs.float()
    positive_ranks[positive_ranks == 0] = float('inf')  # 将非正样本的排名设为无穷大

    # 获取每个查询的最小排名（即最好的正样本排名）
    min_ranks = positive_ranks.min(dim=1)[0]
    min_ranks = min_ranks[min_ranks != float('inf')]  # 过滤掉没有正样本的查询

    # 计算中位数
    return min_ranks.median().item()




def evaluate_model_with_similarity(dataloader, args, scores, recall_k_list=[1, 5, 10]):
    '''

    Args:
        scores: score_matrix_t2i

    Returns:

    '''

    texts_image_index = []
    dataloader = dataloader_with_indices(dataloader)

    # Iterate over the dataloader
    for batch_images, batch_texts, inds in tqdm(dataloader):
        batch_texts_image_index = [ind for ind, texts in zip(inds, batch_texts) for text in texts]

        texts_image_index.extend(batch_texts_image_index)

    # Compute the scores
    positive_pairs = torch.zeros_like(scores, dtype=bool)
    positive_pairs[torch.arange(len(scores)), texts_image_index] = True
    metrics = {}

    # Compute the recall@k
    for recall_k in recall_k_list:
        metrics[f"image_retrieval_recall@{recall_k}"] = (
                batchify(recall_at_k, scores, positive_pairs, args.batch_size, args.device,
                         k=recall_k) > 0).float().mean().item()
        metrics[f"text_retrieval_recall@{recall_k}"] = (
                batchify(recall_at_k, scores.T, positive_pairs.T, args.batch_size, args.device,
                         k=recall_k) > 0).float().mean().item()

    # Compute median rank (medR)
    # 如果数据集较大，可以考虑使用batchify来计算median rank
    metrics["image_retrieval_medR"] = median_rank(scores, positive_pairs)
    metrics["text_retrieval_medR"] = median_rank(scores.T, positive_pairs.T)

    return metrics

def dataloader_with_indices(dataloader):
    """
    Yields batches of data with indices.
    Args:
        dataloader: torch.utils.data.DataLoader
            The dataloader to iterate over.
    Yields:
        x: torch.Tensor
            The input data.
        y: torch.Tensor
            The target data.
        inds: torch.Tensor
            The indices matching y to x.
    """
    start = 0
    for batch in dataloader:
        end = start + len(batch[0])
        inds = torch.arange(start, end)
        yield (*batch, inds)
        start = end

def recall_at_k(scores, positive_pairs, k):
    """
    Computes recall@k for a given set of scores and positive pairs.
    Args:
        scores: torch.Tensor
            The scores of the model.
        positive_pairs: torch.Tensor
            A binary tensor indicating positive pairs.
        k: int
            The value of k for recall@k.
    Returns:
        recall_at_k: torch.Tensor
            The recall@k value.
    """
    nb_texts, nb_images = scores.shape
    topk_indices = torch.topk(scores, k, dim=1)[1]
    nb_positive = positive_pairs.sum(dim=1)
    topk_indices_onehot = torch.nn.functional.one_hot(topk_indices, num_classes=nb_images)
    positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images)
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1,2))
    recall_at_k = (nb_true_positive / nb_positive)
    return recall_at_k


def batchify(func, X, Y, batch_size, device, *args, **kwargs):
    """
    Applies a function to batches of data.
    Args:
        func: callable
            The function to apply.
        X: torch.Tensor
            The input data.
        Y: torch.Tensor
            The target data.
        batch_size: int
            The batch size.
        device: torch.device
            The device to use.
        *args: list
            Additional positional arguments to pass to func.
        **kwargs: dict
            Additional keyword arguments to pass to func.
    Returns:
        results: torch.Tensor
            The results of applying func to the data.
    """
    results = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        x = X[start:end].to(device)
        y = Y[start:end].to(device)
        result = func(x, y, *args, **kwargs).cpu()
        results.append(result)
    return torch.cat(results)