import re
import json
import os

import torch
import torch.distributed as dist


# import utils

# def pre_caption(caption,max_words=50):
#     caption = re.sub(
#         r"([.!\"()*#:;~])",
#         ' ',
#         caption.lower(),
#     )
#     caption = re.sub(
#         r"\s{2,}",
#         ' ',
#         caption,
#     )
#     caption = caption.rstrip('\n')
#     caption = caption.strip(' ')

#     #truncate caption
#     caption_words = caption.split(' ')
#     if len(caption_words)>max_words:
#         caption = ' '.join(caption_words[:max_words])

#     return caption

def pre_caption(caption, max_words=50):
    if isinstance(caption, str):  # Check if caption is a string
        # caption = re.sub(r"([.!\"()*#:;~])", ' ', caption.lower())
        # caption = re.sub(r"\s{2,}", ' ', caption)
        # caption = caption.rstrip('\n')
        # caption = caption.strip(' ')
        # caption_words = caption.split(' ')

        # if len(caption_words) > max_words:
        #     caption = ' '.join(caption_words[:max_words])

        caption = re.sub(
            r"([.!\"()*#:;~])",
            ' ',
            caption.lower(),
        )
        caption = re.sub(
            r"\s{2,}",
            ' ',
            caption,
        )
        caption = caption.rstrip('\n')
        caption = caption.strip(' ')

        # truncate caption
        caption_words = caption.split(' ')
        if len(caption_words) > max_words:
            caption = ' '.join(caption_words[:max_words])
    elif isinstance(caption, list):  # Check if caption is a list
        caption = ' '.join(caption)
        caption = caption.lower()
        caption = re.sub(r"([.!\"()*#:;~])", ' ', caption)
        caption = re.sub(r"\s{2,}", ' ', caption)
        caption = caption.rstrip('\n')
        caption = caption.strip(' ')
        caption_words = caption.split(' ')

        if len(caption_words) > max_words:
            caption = ' '.join(caption_words[:max_words])

    return caption


def pre_question(question, max_ques_words=50):
    question = re.sub(
        r"([.!\"()*#:;~])",
        '',
        question.lower(),
    )
    question = question.rstrip(' ')

    # truncate question
    question_words = question.split(' ')
    if len(question_words) > max_ques_words:
        question = ' '.join(question_words[:max_ques_words])

    return question


