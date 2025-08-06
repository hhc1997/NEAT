import logging

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

from src.open_clip import get_input_dtype, get_tokenizer

def evaluate_model(model, dataloader, args, tokenizer=None, is_synthetic=False):
    """
    Evaluate the model on a multiple-choice question (MCQ) task.

    This function runs the model on a given dataloader, computes the accuracy, and tracks various statistics 
    such as the most common wrong answer types and accuracy per question type. The evaluation logic can be 
    adapted for synthetic datasets by setting the `is_synthetic` flag.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): The dataloader providing the dataset for evaluation.
        args (Namespace): A namespace containing various arguments like device, precision, etc.
        tokenizer (callable, optional): Tokenizer function for processing text inputs. Defaults to None, 
                                        in which case a tokenizer is fetched based on the model.
        is_synthetic (bool, optional): Flag indicating whether the synthetic dataset is used, which changes 
                                       the mapping of wrong answer types. Defaults to False.

    Returns:
        dict: A dictionary containing various metrics such as total accuracy, accuracy by type, most common 
              wrong answer type, wrong answer percentages, predictions by type, and wrong answers by question type.
    """        
    input_dtype = get_input_dtype(args.precision)
    if tokenizer is None:
        tokenizer = get_tokenizer(args.model)

    if "conch" in args.name:
        from conch.open_clip_custom import tokenize

    model.eval()
    total_questions = len(dataloader.dataset)
    correct_answers_sum = 0

    # Initialize dictionaries to keep track of correct answers and total questions by type
    correct_answers_by_type = {'positive': 0, 'negative': 0, 'hybrid': 0}
    total_questions_by_type = {'positive': 0, 'negative': 0, 'hybrid': 0}

    # Map prediction incorrect answer indices to answer types
    if is_synthetic:
        wrong_answer_to_type = {0: 'positive', 1: 'hybrid', 2: 'hybrid', 3: 'negative'}
    else:
        wrong_answer_to_type = {1: 'hybrid', 2: 'positive', 3: 'negative'}

    wrong_answer_counts = {k: 0 for k in wrong_answer_to_type.keys()}

    # Initialize the counter for the number of times each type is predicted
    predictions_by_type = {'positive': 0, 'negative': 0, 'hybrid': 0}

    # Initialize a nested dictionary to map each question type to the number of times each wrong answer type is selected
    wrong_answers_by_question_type = {
        'positive': {'positive': 0, 'negative': 0, 'hybrid': 0},
        'negative': {'positive': 0, 'negative': 0, 'hybrid': 0},
        'hybrid': {'positive': 0, 'negative': 0, 'hybrid': 0}
    }
    # all_prediction = torch.zeros(len(dataloader.dataset))
    # start_idx = 0
    with torch.no_grad():
        for image_tensor, captions, correct_answer, correct_answer_type in tqdm(dataloader, unit_scale=args.batch_size):
            batch_size, num_options = image_tensor.size(0), len(captions)

            image_tensor = image_tensor.to(device=args.device, dtype=input_dtype)
            correct_answer = correct_answer.to(args.device)
            if "conch" in args.name:
                captions_tokens = [tokenize(texts=caption, tokenizer=tokenizer) for caption in captions]
                # Flatten tokens for encoding
                captions_tokens = torch.cat(captions_tokens).to(args.device)
            elif "blip" in args.name:
                captions_tokens = None
            else:
                captions_tokens = [tokenizer(caption) for caption in captions]
                # Flatten tokens for encoding
                captions_tokens = torch.cat(captions_tokens).to(args.device)

            image_features = F.normalize(model.encode_image(image_tensor), dim=-1)
            if "blip" in args.name:
                flat_captions = []
                for caption in captions:
                    flat_captions.extend(caption)
                text_features = model.encode_text(flat_captions, normalize = True)
            else:
                text_features = F.normalize(model.encode_text(captions_tokens), dim=-1)

            # Reshape text features back
            text_features = text_features.view(num_options, batch_size, -1)

            # Compute logits w.r.t. corresponding choices
            logits = torch.einsum('bf,nbf->bn', image_features, text_features)

            predicted_answer = torch.argmax(logits, dim=1)

            # end_idx = start_idx + batch_size
            # all_prediction[start_idx:end_idx] = predicted_answer
            # start_idx = end_idx

            correct_predictions = (predicted_answer == correct_answer).sum().item()
            correct_answers_sum += correct_predictions

            # Update counts for each answer type and track predictions
            for i in range(batch_size):
                answer_type = correct_answer_type[i]
                total_questions_by_type[answer_type] += 1
                if predicted_answer[i] == correct_answer[i]:
                    correct_answers_by_type[answer_type] += 1
                    predictions_by_type[answer_type] += 1
                else:
                    wrong_answer_type = wrong_answer_to_type[predicted_answer[i].item()]
                    wrong_answer_counts[predicted_answer[i].item()] += 1
                    predictions_by_type[wrong_answer_type] += 1
                    wrong_answers_by_question_type[answer_type][wrong_answer_type] += 1
    # torch.save(all_prediction, '/mnt/hanhc/negbench-main/examples/mcq/TCR.pt')
    # Compute overall accuracy
    total_accuracy = correct_answers_sum / total_questions

    # Compute accuracy per type
    # if no questions of this type, the accuracy is meaningless, so we set it to nan
    positive_accuracy = correct_answers_by_type['positive'] / total_questions_by_type['positive'] if total_questions_by_type['positive'] > 0 else float('nan')
    negative_accuracy = correct_answers_by_type['negative'] / total_questions_by_type['negative'] if total_questions_by_type['negative'] > 0 else float('nan')
    hybrid_accuracy = correct_answers_by_type['hybrid'] / total_questions_by_type['hybrid'] if total_questions_by_type['hybrid'] > 0 else float('nan')


    # Compute the most common wrong answer type
    most_common_wrong_answer_type = max(wrong_answer_counts, key=wrong_answer_counts.get)

    # Compute total number of wrong answers and the percentage of each error type
    total_wrong_answers = sum(wrong_answer_counts.values())
    wrong_answer_percentages = {wrong_answer_to_type[k]: (v / total_wrong_answers) * 100 for k, v in wrong_answer_counts.items()}

    # Return a dictionary with all computed metrics
    return {
        'total_accuracy': total_accuracy,
        'positive_accuracy': positive_accuracy,
        'negative_accuracy': negative_accuracy,
        'hybrid_accuracy': hybrid_accuracy,
        'most_common_wrong_answer_type': wrong_answer_to_type[most_common_wrong_answer_type],
        'wrong_answer_percentages': wrong_answer_percentages,
        'predictions_by_type': predictions_by_type,
        'wrong_answers_by_question_type': wrong_answers_by_question_type
    }


def evaluate_binary_mcq_model(model, dataloader, args, tokenizer=None):
    """
    Evaluate the model on a binary multiple-choice question (MCQ) task.

    This function runs the model on a given dataloader, computes the accuracy, and returns total accuracy.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): The dataloader providing the dataset for evaluation.
        args (Namespace): A namespace containing various arguments like device, precision, etc.
        tokenizer (callable, optional): Tokenizer function for processing text inputs. Defaults to None,
                                        in which case a tokenizer is fetched based on the model.

    Returns:
        dict: A dictionary containing the total accuracy.
    """
    input_dtype = get_input_dtype(args.precision)

    if tokenizer is None:
        tokenizer = get_tokenizer(args.model)

    if "conch" in args.name:
        from conch.open_clip_custom import tokenize

    model.eval()
    total_questions = len(dataloader.dataset)
    correct_answers_sum = 0

    with torch.no_grad():
        for image_tensor, captions, correct_answer in tqdm(dataloader, unit_scale=args.batch_size):
            batch_size = image_tensor.size(0)

            # Move inputs to the appropriate device
            image_tensor = image_tensor.to(device=args.device, dtype=input_dtype)
            correct_answer = correct_answer.to(args.device)

            # Tokenize the two captions (caption_0 and caption_1)
            if "conch" in args.name:
                captions_tokens = [tokenize(texts=caption, tokenizer=tokenizer) for caption in captions]
            else:
                captions_tokens = [tokenizer(caption) for caption in captions]

            # Flatten tokens for encoding
            captions_tokens = torch.cat(captions_tokens).to(args.device)

            # Encode image and text features
            image_features = F.normalize(model.encode_image(image_tensor), dim=-1)
            text_features = F.normalize(model.encode_text(captions_tokens), dim=-1)

            # Reshape text features back to handle two choices
            text_features = text_features.view(2, batch_size, -1)

            # Compute logits between the image and the two text captions
            logits = torch.einsum('bf,nbf->bn', image_features, text_features)

            # Predict the answer (either 0 or 1)
            predicted_answer = torch.argmax(logits, dim=1)

            # Count correct predictions
            correct_predictions = (predicted_answer == correct_answer).sum().item()
            correct_answers_sum += correct_predictions

    # Compute overall accuracy
    total_accuracy = correct_answers_sum / total_questions

    # Return the total accuracy
    return {
        'total_accuracy': total_accuracy
    }


def evaluate_binary_mcq_model_hf(model, dataloader, args, tokenizer=None):
    """
    使用 Hugging Face CLIP 模型评估二分类 MCQ 任务

    Args:
        model: Hugging Face CLIPModel
        processor: Hugging Face CLIPProcessor
        dataloader: 数据加载器
        args: 参数
    """
    model.eval()
    model = model.to(args.device)

    total_questions = len(dataloader.dataset)
    correct_answers_sum = 0
    processor = CLIPProcessor.from_pretrained("vinid/plip")
    # 获取输入数据类型
    input_dtype = torch.float16 if args.precision == "fp16" else torch.float32

    with torch.no_grad():
        for image_tensor, captions, correct_answer in tqdm(dataloader, unit_scale=args.batch_size):
            batch_size = image_tensor.size(0)

            # 移动到设备
            image_tensor = image_tensor.to(device=args.device, dtype=input_dtype)
            correct_answer = correct_answer.to(args.device)

            # 处理图像特征
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)

            # 提取图像特征
            image_features = model.get_image_features(pixel_values=image_tensor)
            image_features = F.normalize(image_features, dim=-1)

            # 处理文本 - captions 应该是 [caption_0_list, caption_1_list]
            # 我们需要为每个批次中的每个样本选择对应的标题
            all_texts = []
            for i in range(batch_size):
                # 假设每个类别只用第一个描述，或者你可以随机选择
                all_texts.append(captions[0][i % len(captions[0])])  # 类别0的文本
                all_texts.append(captions[1][i % len(captions[1])])  # 类别1的文本
                # all_texts.append(captions[1][i % len(captions[2])])  # 类别2的文本
                # all_texts.append(captions[1][i % len(captions[3])])  # 类别3的文本


            # 使用 processor 处理文本
            text_inputs = processor(text=all_texts, return_tensors="pt", padding=True)
            text_inputs = {k: v.to(args.device) for k, v in text_inputs.items()}

            # 提取文本特征
            text_features = model.get_text_features(**text_inputs)
            text_features = F.normalize(text_features, dim=-1)

            # 重塑文本特征: [batch_size*2, dim] -> [batch_size, 2, dim]
            text_features = text_features.view(batch_size, 2, -1)

            # 计算相似度分数
            # image_features: [batch_size, dim]
            # text_features: [batch_size, 2, dim]
            logits = torch.einsum('bd,bnd->bn', image_features, text_features)

            # 预测答案
            predicted_answer = torch.argmax(logits, dim=1)

            # 统计正确预测
            correct_predictions = (predicted_answer == correct_answer).sum().item()
            correct_answers_sum += correct_predictions

    # 计算总体准确率
    total_accuracy = correct_answers_sum / total_questions

    return {
        'total_accuracy': total_accuracy
    }

def mcq_eval(model, data, epoch, args, tokenizer=None):
    """
    Evaluate the model across multiple datasets on multiple-choice question (MCQ) tasks.

    This function iterates through different datasets provided in the `data` dictionary, evaluating the model 
    on each dataset using the `evaluate_model` function. Results are aggregated and returned in a dictionary.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data (dict): A dictionary containing datasets as keys and corresponding dataloaders as values.
        epoch (int): The current epoch number, used for logging and tracking.
        args (Namespace): A namespace containing various arguments like device, precision, etc.
        tokenizer (callable, optional): Tokenizer function for processing text inputs. Defaults to None, 
                                        in which case a tokenizer is fetched based on the model.

    Returns:
        dict: A dictionary containing evaluation metrics for each dataset, keyed by dataset name.
    """
    results = {}
    # TODO: specify some frequency of eval in args

    if args.video:
        if 'msrvtt-mcq' in data:
            logging.info('Evaluating on the MSR-VTT MCQ task')
            msrvtt_mcq = evaluate_model(model, data['msrvtt-mcq'].dataloader, args, tokenizer=tokenizer)
            results['msrvtt-mcq-total_accuracy'] = msrvtt_mcq['total_accuracy']
            results['msrvtt-mcq-positive_accuracy'] = msrvtt_mcq['positive_accuracy']
            results['msrvtt-mcq-negative_accuracy'] = msrvtt_mcq['negative_accuracy']
            results['msrvtt-mcq-hybrid_accuracy'] = msrvtt_mcq['hybrid_accuracy']
            results['msrvtt-mcq-most_common_wrong_answer_type'] = msrvtt_mcq['most_common_wrong_answer_type']
            results['msrvtt-mcq-wrong_answer_percentages'] = list(msrvtt_mcq['wrong_answer_percentages'].items())
            results['msrvtt-mcq-predictions_by_type'] = msrvtt_mcq['predictions_by_type']
            results['msrvtt-mcq-wrong_answers_by_question_type'] = msrvtt_mcq['wrong_answers_by_question_type']

    elif args.cxr_dataset:
        if 'chexpert-binary-mcq' in data:
            logging.info('Evaluating on the CheXpert Binary MCQ task')
            chexpert_binary_mcq = evaluate_binary_mcq_model_hf(model, data['chexpert-binary-mcq'].dataloader, args, tokenizer=tokenizer)
            results['chexpert-binary-mcq-total_accuracy'] = chexpert_binary_mcq['total_accuracy']

        if 'chexpert-affirmation-binary-mcq' in data:
            logging.info('Evaluating on the CheXpert Binary Affirmation MCQ task')
            chexpert_binary_affirmation_mcq = evaluate_binary_mcq_model_hf(model, data['chexpert-affirmation-binary-mcq'].dataloader, args, tokenizer=tokenizer)
            results['chexpert-affirmation-binary-mcq-total_accuracy'] = chexpert_binary_affirmation_mcq['total_accuracy']
        # if 'chexpert-mcq' in data:
        #     logging.info('Evaluating on the CheXpert MCQ task')
        #     chexpert_mcq = evaluate_binary_mcq_model_hf(model, data['chexpert-mcq'].dataloader, args, tokenizer=tokenizer)
        #     results['chexpert-mcq-total_accuracy'] = chexpert_mcq['total_accuracy']



    else:
        if 'synthetic-mcq' in data:
            logging.info('Evaluating on the Synthetic MCQ task')
            synthetic_mcq = evaluate_model(model, data['synthetic-mcq'].dataloader, args, tokenizer=tokenizer, is_synthetic=True)
            results['synthetic-mcq-total_accuracy'] = synthetic_mcq['total_accuracy']
            results['synthetic-mcq-positive_accuracy'] = synthetic_mcq['positive_accuracy']
            results['synthetic-mcq-negative_accuracy'] = synthetic_mcq['negative_accuracy']
            results['synthetic-mcq-hybrid_accuracy'] = synthetic_mcq['hybrid_accuracy']
            results['synthetic-mcq-most_common_wrong_answer_type'] = synthetic_mcq['most_common_wrong_answer_type']
            results['synthetic-mcq-wrong_answer_percentages'] = list(synthetic_mcq['wrong_answer_percentages'].items())
            results['synthetic-mcq-predictions_by_type'] = synthetic_mcq['predictions_by_type']
            results['synthetic-mcq-wrong_answers_by_question_type'] = synthetic_mcq['wrong_answers_by_question_type']

        if 'coco-mcq' in data:
            logging.info('Evaluating on the COCO MCQ task')
            coco_mcq = evaluate_model(model, data['coco-mcq'].dataloader, args, tokenizer=tokenizer)
            results['coco-mcq-total_accuracy'] = coco_mcq['total_accuracy']
            results['coco-mcq-positive_accuracy'] = coco_mcq['positive_accuracy']
            results['coco-mcq-negative_accuracy'] = coco_mcq['negative_accuracy']
            results['coco-mcq-hybrid_accuracy'] = coco_mcq['hybrid_accuracy']
            results['coco-mcq-most_common_wrong_answer_type'] = coco_mcq['most_common_wrong_answer_type']
            results['coco-mcq-wrong_answer_percentages'] = list(coco_mcq['wrong_answer_percentages'].items())
            results['coco-mcq-predictions_by_type'] = coco_mcq['predictions_by_type']
            results['coco-mcq-wrong_answers_by_question_type'] = coco_mcq['wrong_answers_by_question_type']

        if 'voc2007-mcq' in data:
            logging.info('Evaluating on the VOC2007 MCQ task')
            voc_mcq = evaluate_model(model, data['voc2007-mcq'].dataloader, args, tokenizer=tokenizer)
            results['voc2007-mcq-total_accuracy'] = voc_mcq['total_accuracy']
            results['voc2007-mcq-positive_accuracy'] = voc_mcq['positive_accuracy']
            results['voc2007-mcq-negative_accuracy'] = voc_mcq['negative_accuracy']
            results['voc2007-mcq-hybrid_accuracy'] = voc_mcq['hybrid_accuracy']
            results['voc2007-mcq-most_common_wrong_answer_type'] = voc_mcq['most_common_wrong_answer_type']
            results['voc2007-mcq-wrong_answer_percentages'] = list(voc_mcq['wrong_answer_percentages'].items())
            results['voc2007-mcq-predictions_by_type'] = voc_mcq['predictions_by_type']
            results['voc2007-mcq-wrong_answers_by_question_type'] = voc_mcq['wrong_answers_by_question_type']



        


    return results