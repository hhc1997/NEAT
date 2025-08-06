import json
import logging
import torch
import wandb
from src.evaluation.mcq import mcq_eval
from src.evaluation.retrieval import retrieval_eval
import os

def evaluate(model, data, epoch, args, score_matrix_t2i = None, tb_writer=None, tokenizer=None):
    """
    Evaluate the model on multiple-choice question (MCQ) and retrieval tasks.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data (dict): A dictionary containing data loaders for evaluation.
        epoch (int): The current epoch number.
        args (argparse.Namespace): Parsed arguments with configurations.
        tb_writer (torch.utils.tensorboard.SummaryWriter, optional): TensorBoard writer for logging metrics.
        tokenizer (callable, optional): Tokenizer function for text inputs.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    print("Evaluating model")
    metrics = {}
    device = torch.device(args.device)
    model.eval()

    print("Evaluating MCQ")
    mcq_metrics = mcq_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(mcq_metrics)

    print("Evaluating Retrieval")
    retrieval_metrics = retrieval_eval(model, data, args, score_matrix_t2i, tokenizer=tokenizer)
    metrics.update(retrieval_metrics)

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch}\n "
        + "\n".join([f"{k}: {round(v, 4):.4f}" if isinstance(v, (int, float)) else f"{k}: {v}" for k, v in metrics.items()])
    )

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        step = None
        if 'train' in data:
            dataloader = data['train'].dataloader
            num_batches_per_epoch = dataloader.num_batches // args.accum_freq
            step = num_batches_per_epoch * epoch
        log_data['epoch'] = epoch
        wandb.log(log_data)

    return metrics


def evaluate_video(model, data, epoch, args, score_matrix_t2i = None, tb_writer=None, tokenizer=None):
    """
    Evaluate the model on video-related tasks, including MCQ and video retrieval.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data (dict): A dictionary containing data loaders for evaluation.
        epoch (int): The current epoch number.
        args (argparse.Namespace): Parsed arguments with configurations.
        tb_writer (torch.utils.tensorboard.SummaryWriter, optional): TensorBoard writer for logging metrics.
        tokenizer (callable, optional): Tokenizer function for text inputs.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    metrics = {}
    device = torch.device(args.device)
    model.eval()

    print("Evaluating MCQ")
    mcq_metrics = mcq_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(mcq_metrics)

    print("Evaluating Video Retrieval")
    retrieval_metrics = retrieval_eval(model, data, args, score_matrix_t2i, tokenizer=tokenizer)
    metrics.update(retrieval_metrics)

    if not metrics:
        raise ValueError("No metrics computed during evaluation.")

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\n".join([f"{k}: {round(v, 4):.4f}" if isinstance(v, (int, float)) else f"{k}: {v}" for k, v in metrics.items()])
    )

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        step = None
        if 'train' in data:
            dataloader = data['train'].dataloader
            num_batches_per_epoch = dataloader.num_batches // args.accum_freq
            step = num_batches_per_epoch * epoch
        log_data['epoch'] = epoch
        wandb.log(log_data)

    return metrics


