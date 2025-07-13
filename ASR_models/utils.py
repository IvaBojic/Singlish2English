import numpy as np
import pandas as pd
from tqdm import tqdm
from torch import autocast
import evaluate as eval_lib
from datasets import Dataset
from bert_score import score
import torch.distributed as dist
from transformers import set_seed
from nltk.tokenize import sent_tokenize
from custom_dataset import AudioTextDataset
import torch, re, wordninja, nltk, warnings, argparse
from torch.utils.data import DataLoader, DistributedSampler
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

warnings.filterwarnings('ignore')
set_seed(30)

# Function called in the end
def cleanup():
    dist.destroy_process_group()

# Normalize text before calculation of the final WER
def normalize_text(sentence, valid_words):

    # Step 1: Remove all words in the words_to_remove list
    words_to_remove = ['a', 'ah', 'aiya', 'an', 'eh', 'er', 'lah', 'leh', 'lor', 
                       'mah', 'oh', 's', 'so', 'the', 'uh', 'um', 'wah', 'wauw', 'wow', 
                       'ya', 'yap', 'yeah', 'yup'
    ]

    pattern_remove = r"\b(?:" + "|".join(words_to_remove) + r")\b"
    sentence = re.sub(pattern_remove, '', sentence, flags=re.IGNORECASE)
    
    # Step 2: Remove all repeated excessive letters
    pattern = r"\b\w*(\w)\1{2,}\w*|\b\w*(\w{2})\2{2,}\w*\b"
    sentence = re.sub(pattern, "", sentence)
    
    # Step 3: Remove any repeating consecutive words (1, 2, or 3-word sequences)
    sentence = re.sub(r"(\b\w+\b)(?:\s+\1\b)+", r"\1", sentence, flags=re.IGNORECASE)  # Remove 1-word repetition
    sentence = re.sub(r"((\b\w+\b\s+){1,2}\b\w+\b)(?:\s+\1\b)+", r"\1", sentence, flags=re.IGNORECASE)  # Remove 2 or 3-word repetition

    # Step 4: Remove words glued together
    new_sentence = ''

    words = re.findall(r"\b\w+\b", sentence)
    for word in words:

        if word in valid_words:
            new_sentence = new_sentence + word + ' '
        else:
            split_words = wordninja.split(word)
            if len(split_words) > 0:
                # if two words are glued together   
                if len(split_words) == 2 and split_words[0] == split_words[1]:
                    new_sentence = new_sentence + split_words[0] + ' '
                elif len(split_words) == 2 and len(split_words[1]) >= len(split_words[0]) and split_words[0] not in valid_words:
                    new_sentence = new_sentence + split_words[1] + ' '
                elif len(split_words) == 2 and len(split_words[0]) >= len(split_words[1]) and split_words[1] not in valid_words:
                    new_sentence = new_sentence + split_words[0] + ' '
                elif split_words[0] == 'mm': 
                    new_sentence = new_sentence + 'm '
                else:
                    new_sentence = new_sentence + word + ' '
            else:
                new_sentence = new_sentence + word + ' '
    
    # Step 5: Normalize contracted forms
    new_sentence = new_sentence.replace(' i m ', ' i am ')
    new_sentence = new_sentence.replace(' i ll ', ' i will ')
    new_sentence = new_sentence.replace(' i d ', ' i would ')
    new_sentence = new_sentence.replace(' i ve ', ' i have ')
    new_sentence = new_sentence.replace(' isn t ', ' is not ')
    new_sentence = new_sentence.replace(' aren t ', ' are not ')
    new_sentence = new_sentence.replace(' wasn t ', ' was not ')
    new_sentence = new_sentence.replace(' weren t ', ' were not ')
    new_sentence = new_sentence.replace(' don t ', ' do not ')
    new_sentence = new_sentence.replace(' doesn t ', ' does not ')
    new_sentence = new_sentence.replace(' didn t ', ' did not ')
    new_sentence = new_sentence.replace(' can t ', ' cannot ')
    new_sentence = new_sentence.replace(' couldn t ', ' could not ')
    new_sentence = new_sentence.replace(' won t ', ' will not ')
    new_sentence = new_sentence.replace(' wouldn t ', ' would not ')
    new_sentence = new_sentence.replace(' shouldn t ', ' should not ')
    new_sentence = new_sentence.replace(' mightn t ', ' might not ')
    new_sentence = new_sentence.replace(' mustn t ', ' must not ')
    new_sentence = new_sentence.replace(' haven t ', ' have not ')
    new_sentence = new_sentence.replace(' you d ', ' you would ')
    new_sentence = new_sentence.replace(' you ll ', ' you will ')
    new_sentence = new_sentence.replace(' you re ', ' you are ')
    new_sentence = new_sentence.replace(' you ve ', ' you have ')
    new_sentence = new_sentence.replace(' he d ', ' he would ')
    new_sentence = new_sentence.replace(' he ll ', ' he will ')
    new_sentence = new_sentence.replace(' he s ', ' he is ')
    new_sentence = new_sentence.replace(' she d ', ' she would ')
    new_sentence = new_sentence.replace(' she ll ', ' she will ')
    new_sentence = new_sentence.replace(' she s ', ' she is ')
    new_sentence = new_sentence.replace(' it d ', ' it would ')
    new_sentence = new_sentence.replace(' it ll ', ' it will ')
    new_sentence = new_sentence.replace(' it s ', ' it is ')
    new_sentence = new_sentence.replace(' we d ', ' we would ')
    new_sentence = new_sentence.replace(' we ll ', ' we will ')
    new_sentence = new_sentence.replace(' we ve ', ' we have ')
    new_sentence = new_sentence.replace(' we re ', ' we were ')
    new_sentence = new_sentence.replace(' they d ', ' they would ')
    new_sentence = new_sentence.replace(' they ll ', ' they will ')
    new_sentence = new_sentence.replace(' they re ', ' they are ')
    new_sentence = new_sentence.replace(' they ve ', ' they have ')
    new_sentence = new_sentence.replace(' who s ', ' who is ')
    new_sentence = new_sentence.replace(' who d ', ' who would ')
    new_sentence = new_sentence.replace(' who ll ', ' who will ')
    new_sentence = new_sentence.replace(' what s ', ' what is ')
    new_sentence = new_sentence.replace(' what ll ', ' what will ')
    new_sentence = new_sentence.replace(' that s ', ' that is ')
    new_sentence = new_sentence.replace(' there s ', ' there is ')
    new_sentence = new_sentence.replace(' here s ', ' here is ')
    new_sentence = new_sentence.replace(' where s ', ' where is ')
    new_sentence = new_sentence.replace(' how s ', ' how is ')
    new_sentence = new_sentence.replace(' let s ', ' let us ')

    # Clean up extra spaces from removal of words
    return re.sub(r"\s+", " ", new_sentence).strip()

# Calculate all transcribtion/translation metrics
def calculate_metrics(label_strs, pred_strs, tokenizer, valid_words, task):
    # For returning all results
    results = {}
    normalizer = BasicTextNormalizer()

    if(tokenizer is not None):
        label_strs = [s.replace('-100', str(tokenizer.pad_token_id)) for s in label_strs]

    if('transcribe' in task): 
        
        # Load the WER metric
        wer_metric = eval_lib.load('wer')
 
        # Compute ortographic WER (Word Error Rate)
        wer_ortho = wer_metric.compute(predictions=pred_strs, references=label_strs) 

        # Compute normalized WER
        pred_str_norm = [normalizer(pred) for pred in pred_strs]
        label_str_norm = [normalizer(label) for label in label_strs]
        pred_str_norm = [pred_str_norm[i] for i in range(len(pred_str_norm)) if len(label_str_norm[i]) > 0]
        label_str_norm = [label_str_norm[i] for i in range(len(label_str_norm)) if len(label_str_norm[i]) > 0]
        wer = wer_metric.compute(predictions=pred_str_norm, references=label_str_norm) 

        # Compute final WER
        pred_str_remove = [normalize_text(pred, valid_words) for pred in pred_str_norm]
        label_str_remove = [normalize_text(label, valid_words) for label in label_str_norm]
        pred_str_remove = [pred_str_remove[i] for i in range(len(pred_str_remove)) if len(label_str_remove[i]) > 0]
        label_str_remove = [label_str_remove[i] for i in range(len(label_str_remove)) if len(label_str_remove[i]) > 0]
        
        # Sentence-level WER calculates WER for each individual sentence separately, then averages the values
        wer_remove_list = [round(wer_metric.compute(predictions=[pred], references=[ref]), 2) for pred, ref in zip(pred_str_remove, label_str_remove)] 
        
        # Corpus-level WER calculates WER across the entire dataset, treating it as one long text.
        #wer_remove_corpus = wer_metric.compute(predictions=pred_str_remove, references=label_str_remove) 

        results['wer_ortho'] = wer_ortho
        results['wer'] = wer
        results['wer_remove'] = np.mean(wer_remove_list) 
        
        results['wer_remove_list'] = wer_remove_list
        results['labels'] = label_str_remove
        results['predictions'] = pred_str_remove

    elif('translat' in task):
        
        pred_strs_tran = [pred.strip() for pred in pred_strs]
        label_strs_tran = [label.strip() for label in label_strs]

        pred_str_norm = [normalizer(pred) for pred in pred_strs]
        label_str_norm = [normalizer(label) for label in label_strs]
        
        pred_strs_tran = [normalize_text(pred, valid_words) for pred in pred_str_norm]
        label_strs_tran = [normalize_text(label, valid_words) for label in label_str_norm]

        # ROUGE expects newline after each sentence
        preds_rouge = ['\n'.join(sent_tokenize(pred)) for pred in pred_strs_tran]
        labels_rouge = ['\n'.join(sent_tokenize(label)) for label in label_strs_tran]

        # Compute ROUGE
        rouge_results = eval_lib.load('rouge').compute(
            predictions=preds_rouge,
            references=labels_rouge,
            use_stemmer=True,
        )
        rouge_results = {f"{k}": round(v, 4) for k, v in rouge_results.items()}  

        # Compute BLEU sentence level
        total_score = 0
        bleu_list = []
        for i in range(len(label_strs_tran)):
            reference = [label_strs_tran[i].split()]
            prediction = pred_strs_tran[i].split()
            bleu_result = nltk.translate.bleu_score.sentence_bleu(reference, 
                                                                prediction, 
                                                                weights = (0.25, 0.25, 0.25, 0.25)
                                                                ) 
            bleu_list.append(round(bleu_result, 4))
            total_score += bleu_result 
        bleu_results = round(total_score / len(label_strs_tran), 4)

        # Calculationg BLEU on the corpus level
        #results = bleu.compute(predictions=pred_strs_tran, references=label_strs_tran)

        # Compute BertScore
        _, _, F1 = score(pred_strs_tran, label_strs_tran, lang='en', verbose=True)
        bertScore = round(F1.mean().item(), 4) 

        results['rouge1'] = rouge_results['rouge1']
        results['rouge2'] = rouge_results['rouge2']
        results['rougeL'] = rouge_results['rougeL']
        results['rougeLsum'] = rouge_results['rougeLsum']
        results['bleu'] = bleu_results
        results['bertScore'] = bertScore
        results['bleu_list'] = bleu_list
        results['labels'] = label_strs_tran
        results['predictions'] = pred_strs_tran
    return results

# Dist function to do inference and evaluate results with GT
def evaluate_dist(rank, valid_loader, device, model, model_name_short, task, tokenizer, forced_decoder_ids, valid_words):
    pred_strs, label_strs = [], []

    # Create a CUDA stream
    stream = torch.cuda.Stream()
    
    if rank == 0: 
        with tqdm(total=len(valid_loader), desc='Evaluating', unit='batch', position=rank) as pbar:
            with torch.no_grad():
                for batch in valid_loader:

                    # Transfer data to GPU asynchronously with the created stream
                    with torch.cuda.stream(stream): 
                        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                    
                    # Wait for the stream to synchronize before computation
                    torch.cuda.synchronize()

                    # Forward pass through the model
                    with autocast(device_type='cuda'):

                        if('t5' in model_name_short):
                            pred_ids = model(input_values=batch['input_features'],
                                    labels=batch['labels'])
                            pred_ids = pred_ids.logits.argmax(dim=-1) 
                        elif('whisper' in model_name_short):
                            pred_ids = model.module.generate(input_features=batch['input_features'],                                    
                                        labels=batch['labels'], forced_decoder_ids=forced_decoder_ids)  
                        else: #('seamless' in model_name_short):
                            pred_ids = model(input_features=batch['input_features'],                                    
                                    labels=batch['labels'])
                            pred_ids = pred_ids.logits.argmax(dim=-1)         

                    # Get predictions
                    pred_strs.extend(tokenizer.batch_decode(pred_ids, skip_special_tokens=True)) 
                    label_strs.extend(tokenizer.batch_decode(batch['labels'], skip_special_tokens=True))

                    # Update progress bar
                    pbar.update(1)
    else:
         with torch.no_grad():
             for batch in valid_loader:

                # Transfer data to GPU asynchronously h the created stream
                with torch.cuda.stream(stream): 
                    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

                # Wait for the stream to synchronize before computation
                torch.cuda.synchronize()       
        
                # Forward pass through the model
                with autocast(device_type='cuda'):

                    if('t5' in model_name_short):
                        pred_ids = model(input_values=batch['input_features'],
                                    labels=batch['labels'])
                        pred_ids = pred_ids.logits.argmax(dim=-1)
                    elif('whisper' in model_name_short):
                        pred_ids = model.module.generate(input_features=batch['input_features'],                                    
                                    labels=batch['labels'], forced_decoder_ids=forced_decoder_ids)  
                    else: #('seamless' in model_name_short):
                        pred_ids = model(input_features=batch['input_features'],                                    
                                    labels=batch['labels'])
                        pred_ids = pred_ids.logits.argmax(dim=-1)

                # Get predictions
                pred_strs.extend(tokenizer.batch_decode(pred_ids, skip_special_tokens=True))  
                label_strs.extend(tokenizer.batch_decode(batch['labels'], skip_special_tokens=True))
    
    results = calculate_metrics(label_strs, pred_strs, tokenizer, valid_words, task)
    
    # Reduce metrics across all processes
    if('transcribe' in task): 
        # Reduce WER ortho
        wer_ortho_tensor = torch.tensor(results['wer_ortho']).to(rank)
        dist.reduce(wer_ortho_tensor, dst=0, op=dist.ReduceOp.SUM)
        results['wer_ortho'] = wer_ortho_tensor.item()
    
        # Reduce WER normalized
        wer_tensor = torch.tensor(results['wer']).to(rank)
        dist.reduce(wer_tensor, dst=0, op=dist.ReduceOp.SUM)
        results['wer'] = wer_tensor.item()
    
        # Reduce WER repeatead
        wer_remove_tensor = torch.tensor(results['wer_remove']).to(rank)
        dist.reduce(wer_remove_tensor, dst=0, op=dist.ReduceOp.SUM)
        results['wer_remove'] = wer_remove_tensor.item()
    elif('translat' in task):
        # Reduce ROUGE1
        rouge1_tensor = torch.tensor(results['rouge1']).to(rank)
        dist.reduce(rouge1_tensor, dst=0, op=dist.ReduceOp.SUM)
        results['rouge1'] = rouge1_tensor.item()

        # Reduce ROUGE2
        rouge2_tensor = torch.tensor(results['rouge2']).to(rank)
        dist.reduce(rouge2_tensor, dst=0, op=dist.ReduceOp.SUM)
        results['rouge2'] = rouge2_tensor.item()

        # Reduce ROUGEL
        rougeL_tensor = torch.tensor(results['rougeL']).to(rank)
        dist.reduce(rougeL_tensor, dst=0, op=dist.ReduceOp.SUM)
        results['rougeL'] = rougeL_tensor.item()

        # Reduce ROUGELsum
        rougeLsum_tensor = torch.tensor(results['rougeLsum']).to(rank)
        dist.reduce(rougeLsum_tensor, dst=0, op=dist.ReduceOp.SUM)
        results['rougeLsum'] = rougeLsum_tensor.item()

        # Reduce BLEU
        bleu_tensor = torch.tensor(results['bleu']).to(rank)
        dist.reduce(bleu_tensor, dst=0, op=dist.ReduceOp.SUM)
        results['bleu'] = bleu_tensor.item()

        # Reduce BertScore
        bertScore_tensor = torch.tensor(results['bertScore']).to(rank)
        dist.reduce(bertScore_tensor, dst=0, op=dist.ReduceOp.SUM)
        results['bertScore'] = bertScore_tensor.item()
    else:
        return {}
    
    # Aggregate results on the master process
    if rank == 0:
        num_processes = dist.get_world_size()
        results = {
            key: value if isinstance(value, list) else value / num_processes
            for key, value in results.items()
        }

        return results
    return {}

# Loading of the data (train/valid set)
def data_load(rank, world_size, train_path, valid_path, tokenizer, 
                feature_extractor, MAX_LENGTH, train_batch, model_name):
    
    # Read train/valid set
    train_df = Dataset.from_pandas(pd.read_csv(train_path))
    valid_df = Dataset.from_pandas(pd.read_csv(valid_path))
    
    # Initialize the custom dataset for training and evaluation
    train_dataset = AudioTextDataset(train_df, feature_extractor, tokenizer, 
                                     MAX_LENGTH, model_name)
    eval_dataset = AudioTextDataset(valid_df, feature_extractor, tokenizer, 
                                    MAX_LENGTH, model_name)
    
    # Use DistributedSampler for distributed data loading
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, 
                                       rank=rank, shuffle=True)
    eval_sampler = DistributedSampler(eval_dataset, num_replicas=world_size, 
                                      rank=rank, shuffle=False)
    
    eval_batch = int(train_batch / 2)

    train_loader = DataLoader(train_dataset, batch_size=train_batch, 
                              num_workers=train_batch, prefetch_factor=train_batch, 
                    persistent_workers=True, sampler=train_sampler, pin_memory=True)
    
    eval_loader = DataLoader(eval_dataset, batch_size=eval_batch, 
                             num_workers=eval_batch, prefetch_factor=eval_batch, 
                    persistent_workers=True, sampler=eval_sampler, pin_memory=True)
    
    if rank == 0:
        print(f"train/valid: {len(train_df)}/{len(valid_df)}")

    return train_loader, eval_loader

# Loading of the data (test)
def data_load_test(rank, world_size, test_file, tokenizer, 
                feature_extractor, MAX_LENGTH, test_batch, model_name):
    
    # Read test set
    test_df = Dataset.from_pandas(pd.read_csv(test_file))
    
    # Initialize the custom dataset for training and evaluation
    test_dataset = AudioTextDataset(test_df, feature_extractor, tokenizer, MAX_LENGTH, model_name)
    
    # Use DistributedSampler for distributed data loading
    train_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    test_loader = DataLoader(test_dataset, batch_size=test_batch, num_workers=test_batch, prefetch_factor=test_batch, 
                    persistent_workers=True, sampler=train_sampler, pin_memory=True)
    
    if rank == 0:
        print(f"test: {len(test_df)}")

    return test_loader

# Restrict integer values for potive numbers
def check_positive(value):
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError("The argument must be a negative integer.")
    return ivalue

# Restrict integer values for --word_size
def restricted_world_size(value):
    ivalue = int(value)
    if ivalue < 0 or ivalue > 8:
        raise argparse.ArgumentTypeError("The second parameter must be between 0 and 8.")
    return ivalue

# Restrict integer values for --model
def restricted_models(value):
    ivalue = int(value)
    if ivalue < 0 or ivalue > 3:
        raise argparse.ArgumentTypeError("The second parameter must be between 0 and 3.")
    return ivalue

# Seting the world size
def setup(rank, world_size):
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
