from tqdm import tqdm
from torch import autocast
from torch import GradScaler
from torch.optim import AdamW
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.parallel import DistributedDataParallel as DDP
import logging, nltk, utils, warnings, os, torch, shutil, time, argparse
from transformers import set_seed, AutoTokenizer, AutoFeatureExtractor, WhisperProcessor, WhisperTokenizer, WhisperFeatureExtractor, WhisperForConditionalGeneration, SeamlessM4TModel, SpeechT5ForSpeechToText, SpeechT5Tokenizer, SpeechT5Processor, SpeechT5ForSpeechToText, SpeechT5Tokenizer, SpeechT5Processor

warnings.filterwarnings('ignore')
set_seed(30)

# Current score (WER/BLEU) is propagated across all processes
def step_scheduler_and_broadcast(rank, plateau_scheduler, current_score):
    
    # Step the scheduler only on rank 0
    if rank == 0:
        plateau_scheduler.step(current_score)
    
    # Create a state dictionary for broadcasting
    scheduler_state = plateau_scheduler.state_dict()

    # Convert the dictionary to a list for broadcasting
    state_list = [scheduler_state]
    
    # Broadcast the scheduler state from rank 0 to all other processes
    if rank == 0:
        dist.broadcast_object_list(state_list, src=0)
    else:
        dist.broadcast_object_list(state_list, src=0)
        scheduler_state = state_list[0]
    
    # Load the broadcasted state into the scheduler for other ranks
    plateau_scheduler.load_state_dict(scheduler_state)

    return plateau_scheduler
 
 # One step of the epoch

# Loop for fine-tuning
def loop(rank, world_size, train_file, valid_file, model, model_name, 
                model_name_short, task, tokenizer, feature_extractor, forced_decoder_ids,
                MAX_LENGTH, train_batch, learning_rate, num_epochs, valid_steps, 
                accumulation_batches, valid_words, output_dir, output_file
    ):
    
    # Load data
    train_loader, valid_loader = utils.data_load(
                                    rank, world_size, train_file, valid_file, 
                                    tokenizer, feature_extractor, MAX_LENGTH, 
                                    train_batch, model_name
                                )

    # Initialize optimizer and scheduler
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                      lr=learning_rate, weight_decay=0.01)
    
    # Define scheduler parameters
    if ('transcribe' in task):
        best_score = 100
        plateau_scheduler = ReduceLROnPlateau(optimizer, mode='min', 
                                              factor=0.5, patience=2, 
                                              threshold = 0.01) 
    elif('translat' in task):
        best_score = 0
        plateau_scheduler = ReduceLROnPlateau(optimizer, mode='max', 
                                              factor=0.5, patience=2, 
                                              threshold = 0.01) 

    best_model_path = ''
    scaler = GradScaler()
    stream = torch.cuda.Stream()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    for epoch in range(num_epochs):
        model.train()
        train_loader.sampler.set_epoch(epoch) 

        if rank == 0:
            # Initializing values for process 0
            start_time = time.time()    

            with open(output_file, 'a') as file:
                with tqdm(total=len(train_loader), desc=f"Training epoch {epoch}/{num_epochs - 1}") as pbar:
                    for batch_idx, batch in enumerate(train_loader):

                        # Transfer data to GPU asynchronously with the created stream
                        with torch.cuda.stream(stream): 
                            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                                
                        # Forward pass through the model
                        with autocast(device_type=f"cuda" if torch.cuda.is_available() else "cpu"): 

                            if('t5' in model_name_short):
                                outputs = model(
                                    input_values=batch['input_features'],
                                    labels=batch['labels']
                                )
                            else:
                                 outputs = model(
                                    input_features=batch['input_features'],                                    
                                    labels=batch['labels']
                                )
                            loss = outputs.loss.mean() / accumulation_batches  # Scale the loss by accumulation batches                            
                        
                        # Wait for the stream to synchronize before computation
                        torch.cuda.synchronize()

                        # Backward pass
                        scaler.scale(loss).backward()
                        
                        # Update model parameters every accumulation_steps
                        if (batch_idx + 1) % accumulation_batches == 0:
                            # Perform optimizer step
                            scaler.step(optimizer)
                            scaler.update()
                            
                            # Clip gradients
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            
                            # Zero gradients
                            optimizer.zero_grad()
                    
                        # Update progress bar
                        pbar.update(1)

                        # Perform validation
                        if (batch_idx + 1) % valid_steps == 0:
                            model.eval()
                            eval_results = utils.evaluate_dist(rank, valid_loader, device,
                                                        model, model_name_short, task,
                                                        tokenizer, forced_decoder_ids, valid_words)

                            # Write in a file
                            file.write(f"Epoch: {epoch}, {batch_idx + 1}/{len(train_loader)}: ")
                            file.write(f"train_loss={loss.item():.4f}, ") 
                            file.write(f"lr={optimizer.param_groups[0]['lr']:.7f}, ")     
                            
                            if ('transcribe' in task):      
                                file.write(f"wer_ortho={round(eval_results['wer_ortho'] * 100, 2)}, ") 
                                file.write(f"wer={round(eval_results['wer']* 100, 2)}, ") 
                                file.write(f"wer_remove={round(eval_results['wer_remove']* 100, 2)}\n") 

                                # We are using WER (normalized) as our metric 
                                # in validation for transcription
                                current_score = eval_results['wer']

                            elif('translat' in task):
                                file.write(f"rouge1={round(eval_results['rouge1'], 4)}, ") 
                                file.write(f"rouge2={round(eval_results['rouge2'], 4)}, ") 
                                file.write(f"rougeL={round(eval_results['rougeL'], 4)}, ") 
                                file.write(f"rougeLsum={round(eval_results['rougeLsum'], 4)}, ") 
                                file.write(f"bleu={round(eval_results['bleu'], 4)}, ") 
                                file.write(f"bertscore={round(eval_results['bertScore'], 4)}\n") 

                                # We are using bleu as our metric 
                                # in validation for transcription
                                current_score = eval_results['bleu']
                            file.flush()   

                            # Update the best score if needed and save the best model
                            if (('transcribe' in task and current_score < best_score) or 
                                ('translat' in task and current_score > best_score)): 
                                
                                best_score = current_score 

                                # Save the best model
                                model_path = os.path.join(output_dir, f"{epoch}_{batch_idx + 1}")
                                model.module.save_pretrained(model_path)
                                feature_extractor.save_pretrained(model_path)
                                tokenizer.save_pretrained(model_path)

                                file.write(f"Saving model checkpoint to {model_path}\n")
                                file.flush()

                                # Delete previous checkpoints
                                if os.path.exists(best_model_path):
                                    shutil.rmtree(best_model_path)
                                best_model_path = model_path

                            # Synchronize all processes before updating the scheduler or unfreezing layers
                            dist.barrier() 

                            # Step the scheduler based on evaluation loss and unfreeze layers if necessary
                            plateau_scheduler = step_scheduler_and_broadcast(rank, plateau_scheduler,  
                                                                             current_score) 
                            
                            model.train()

                    elapsed_time = (time.time() - start_time) / 3600
                    file.write(f"Time taken for epoch {epoch}: {elapsed_time:.2f} hours\n")
                    file.flush()
        else:
            for batch_idx, batch in enumerate(train_loader):
                
                # Transfer data to GPU asynchronously with the created stream
                with torch.cuda.stream(stream): 
                    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                        
                # Forward pass through the model
                with autocast(device_type=f"cuda" if torch.cuda.is_available() else "cpu"): 

                    if('t5' in model_name_short):
                        outputs = model(input_values=batch['input_features'],
                                    labels=batch['labels'])
                    else:
                        outputs = model(input_features=batch['input_features'],                                    
                                    labels=batch['labels'])
                    loss = outputs.loss.mean() / accumulation_batches  # Scale the loss by accumulation batches

                # Wait for the stream to synchronize before computation
                torch.cuda.synchronize()

                # Backward pass
                scaler.scale(loss).backward()

                # Update model parameters every accumulation_steps
                if (batch_idx + 1) % accumulation_batches == 0:
                    # Perform optimizer step
                    scaler.step(optimizer)
                    scaler.update()
                    
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # Zero gradients
                    optimizer.zero_grad()

                # Perform evaluation
                if (batch_idx + 1) % valid_steps == 0:
                    model.eval()
                    utils.evaluate_dist(rank, valid_loader, device, model, model_name_short,
                                         task, tokenizer, forced_decoder_ids, valid_words)
                    
                    # Synchronize all processes before updating the scheduler or unfreezing layers
                    dist.barrier() 

                    # Step the scheduler based on evaluation loss and unfreeze layers if necessary
                    plateau_scheduler = step_scheduler_and_broadcast(rank, plateau_scheduler,  
                                                                     best_score) 
                    
                    model.train()

# Function for transcription/translation fine-tuning 
def train(rank, world_size, train_file, valid_file, model_name, model_name_short, language, 
          task, MAX_LENGTH, train_batch, learning_rate, num_epochs, valid_steps, 
          accumulation_batches, valid_words, output_dir, output_file):
    
    logging.getLogger('transformers').setLevel(logging.ERROR)
    utils.setup(rank, world_size)
    
    # Create model/tokenizer/feature_extractor
    if('whisper' in model_name_short):
        tokenizer = WhisperTokenizer.from_pretrained(model_name, language=language, task=task)
        feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name, language=language, task=task)  
        model = WhisperForConditionalGeneration.from_pretrained(model_name).to(rank)
        processor = WhisperProcessor.from_pretrained(model_name, language=language, task=task)  
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task) 
    elif('seamless' in model_name_short): 
        tokenizer = AutoTokenizer.from_pretrained(model_name, language=language, task=task)   
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, language=language, task=task)
        model = SeamlessM4TModel.from_pretrained(model_name).to(rank)
        forced_decoder_ids = None
    elif('t5' in model_name_short):
        tokenizer = SpeechT5Tokenizer.from_pretrained(model_name, language=language)
        feature_extractor = SpeechT5Processor.from_pretrained(model_name, language=language)
        model = SpeechT5ForSpeechToText.from_pretrained(model_name, ignore_mismatched_sizes=True).to(rank)
        forced_decoder_ids = None
    else:
        return
    
    # Prepare model for distributed fine-tuning     
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    model.module.config.pad_token_id = model.module.config.eos_token_id + 1
   
    try:
        loop(
            rank, world_size, train_file, valid_file, model, model_name, model_name_short, 
            task, tokenizer, feature_extractor, forced_decoder_ids, MAX_LENGTH, train_batch,
            learning_rate, num_epochs, valid_steps, accumulation_batches, valid_words, 
            output_dir, output_file
        )
    finally:
        utils.cleanup()

def main():

    # Model names [use index for args.model parameter]
    model_names = [ 
        'openai/whisper-small',
        'openai/whisper-medium',
        'facebook/hf-seamless-m4t-medium',
        'microsoft/speecht5_asr'
    ]

    valid_words = set(nltk.corpus.words.words())

    parser = argparse.ArgumentParser(description="Transcription/translation models fine-tuning")

    # Folder arguments with defaults
    parser.add_argument('--base_folder', default='output',
                        help="Base folder to store fine-tuned models and outputs.")
    parser.add_argument('--base_dataset', default='../small_dataset',
                        help="Base folder for transcription/translation datasets.")

    # Training configuration arguments
    parser.add_argument('--world_size', type=utils.restricted_world_size, default=1,
                        help="Number of GPUs for fine-tuning (default: 1)")
    parser.add_argument('--task', choices=['transcribe', 'translate'], default='transcribe',
                        help="Task type: transcribe or translate (default: transcribe)")
    parser.add_argument('--model', type=utils.restricted_models, default=0,
                        help="Model index: 0-3 (default: 0 (whisper_small))")
    parser.add_argument('--batch', type=utils.check_positive, default=4,
                        help="Batch size for fine-tuning (default: 4)")
    parser.add_argument('--accumulation', type=utils.check_positive, default=4,
                        help="Gradient accumulation steps (default: 4)")
    parser.add_argument('--epoch', type=utils.check_positive, default=5,
                        help="Number of epochs (default: 5)")
    parser.add_argument('--steps', type=utils.check_positive, default=5,
                        help="Validation steps (default: 5)")
    parser.add_argument('--max_length', type=int, default=448,
                        help="Max token length for fine-tuning (default: 448)")
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help="Learning rate for fine-tuning (default: 1e-5)")
    parser.add_argument('--language', default='English',
                        help="Language for fine-tuning (default: English)")

    args = parser.parse_args()

    # Unpack arguments
    base_folder = args.base_folder
    base_dataset = args.base_dataset
    world_size = args.world_size
    task = args.task
    model_name = model_names[args.model]
    train_batch = args.batch
    accumulation_batches = args.accumulation
    num_epochs = args.epoch
    valid_steps = args.steps
    language = args.language
    MAX_LENGTH = args.max_length
    learning_rate = args.learning_rate

    model_name_short = model_name.lower().split('/')[-1]
    print(f"Using model: {model_name_short}")

    # Determine files and output structure based on task
    train_file = f"{base_dataset}/train_{task}.csv"
    valid_file = f"{base_dataset}/valid_{task}.csv"
    
    output_dir = f"{base_folder}/{task}/{model_name_short}"
    output_file = f"{base_folder}/{task}/{model_name_short}.eval"

    # Adjust language for seamless model in translation
    if 'seamless' in model_name_short:
        language = 'en'
        if 'translate' in task:
            task = 'speech_to_text_translation'

    # Start fine-tuning
    mp.spawn(train, args=(
        world_size, train_file, valid_file, model_name, model_name_short,
        language, task, MAX_LENGTH, train_batch, learning_rate,
        num_epochs, valid_steps, accumulation_batches, valid_words,
        output_dir, output_file
    ),
    nprocs=world_size,
    join=True)

if __name__ == '__main__':

   main()