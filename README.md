This repository contains the official PyTorch implementation of the paper: Negation-Aware Test-Time Adaptation for Vision-Language Models.

## Overview
<h1 align="center"><img src="intro.jpg" width="75%"></h1>

> **Negation-Aware Test-Time Adaptation for Vision-Language Models**<br>
> Haochen Han, Alex Jinpeng Wang, Fangming Liu, Jun Zhu.<br>
> [https://www.arxiv.org/abs/2507.19064](https://www.arxiv.org/abs/2507.19064) 
>
> **Abstract:** *In this paper, we study a practical but less-touched problem in Vision-Language Models (VLMs), \ie, negation understanding. Specifically, many real-world applications require models to explicitly identify what is false or non-existent, \eg, radiologists may search for images that exclude specific conditions. Despite the impressive transferability of VLMs through large-scale training, they suffer from a critical limitation that fails to handle negation. To address this challenge, existing methods attribute its root cause to the scarcity of negation training data and propose to fine-tune VLMs on massive data containing explicit negation. Undoubtedly, such data-centric solutions demand substantial data and computational resources, limiting their sustainable widespread adoption. To tackle negation in a low-carbon manner, we empirically observe that the key obstacle lies in the dual-concept shifts between the affirmation and negation distributions. Therefore, we propose a Negation-Aware Test-Time Adaptation (NEAT) method to efficiently adjust distribution-related parameters during inference. In brief, NEAT can reduce distribution shift in consistent semantics while eliminating false distributional consistency in unrelated semantics. Extensive experiments on the various negation understanding tasks verify the effectiveness of the proposed method. Remarkably, with less than 0.01\% of trainable parameters, NEAT achieves comparable or superior performance to state-of-the-art post-training approaches.*

## Dataset Preparation
We use the recent advance NegBench to evaluate models' negation understanding capabilities, which covers images, videos, and medical images. The `data/`  format is as follows:
```
data/
├── images/
│   ├── voc2007 (raw images)
│   ├── val2017 (raw images)
│   ├── COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true_logic_inversion.csv
│   ├── COCO_val_retrieval.csv
│   ├── COCO_val_mcq_llama3.1_rephrased.csv
│   └── VOC2007_mcq_llama3.1_rephrased.csv
├── videos/
│   ├── TestVideo (raw video)
│   ├── msr_vtt_retrieval_rephrased_llama_logic_inversion.csv
|   ├── msr_vtt_retrieval.csv
│   └── msr_vtt_mcq_rephrased_llama.csv
├── chexpert/
│   ├── valid (raw medical images)
│   ├── Atelectasis
│       ├── chexpert_affirmation_binary_mcq.csv
│       └── chexpert_negation_binary_mcq.csv
│   ├── Cardiomegaly
│       ├── chexpert_affirmation_binary_mcq.csv
│       └── chexpert_negation_binary_mcq.csv
│   ├── Consolidation
│       ├── chexpert_affirmation_binary_mcq.csv
│       └── chexpert_negation_binary_mcq.csv
│   ├── Lung_Opacity
│       ├── chexpert_affirmation_binary_mcq.csv
│       └── chexpert_negation_binary_mcq.csv
```
For experimental convenience, we directly provide the LLM-processed captions in `xxx_logic_inversion.csv`. All CSV files and corresponding raw images or videos are available for download:
