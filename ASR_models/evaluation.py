import pandas as pd
from transformers import set_seed
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import time, utils, logging, warnings, nltk, torch, argparse
from transformers import WhisperTokenizer, WhisperProcessor, WhisperFeatureExtractor, WhisperForConditionalGeneration

set_seed(30)
warnings.filterwarnings('ignore')

# Function to do inference and evaluate results with GT
def evaluate(rank, world_size, model_name, model_name_short, test_file, output_file,
             max_input_length, language, task, test_batch, valid_words, save):
    
    logging.getLogger('transformers').setLevel(logging.ERROR)
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    utils.setup(rank, world_size) 
    
    if('whisper' in model_name_short):
        tokenizer = WhisperTokenizer.from_pretrained(model_name, language=language, task=task)
        feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name, language=language, task=task)
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        processor = WhisperProcessor.from_pretrained(model_name, language=language, task=task)  
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task) 
    else:
        return

    model.config.language = language
    model.config.task = task
    
    model = model.to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    model.module.config.pad_token_id = model.module.config.eos_token_id + 1
    
    model.eval() 

    try:
        start_time = time.time()   

        test_loader = utils.data_load_test(rank, world_size, test_file, tokenizer, 
            feature_extractor, max_input_length, test_batch, model_name) 
        eval_results = utils.evaluate_dist(rank, test_loader, device, model, model_name_short,
                                                task, tokenizer, forced_decoder_ids, valid_words) 

        if (rank == 0 and len(eval_results) > 0):
            with open(output_file, 'a') as file:
                if('transcribe' in task):
                    file.write(f"wer_ortho={round(eval_results['wer_ortho'] * 100, 2)}, ")      
                    file.write(f"wer={round(eval_results['wer']* 100, 2)}, ")                  
                    file.write(f"wer_remove={round(eval_results['wer_remove']* 100, 2)}\n")    
                
                    if 'yes' in save:
                        df = pd.DataFrame({
                            'WER_remove': [round(x * 100, 2) for x in eval_results['wer_remove_list']],
                            'Value': eval_results['predictions'],
                            'Label': eval_results['labels'],
                        }) #.sort_values(by='Label', ascending=True)

                        df.to_csv(f"{output_file}_inf", index=False)

                elif('translat' in task):
                    file.write(f"rouge1={round(eval_results['rouge1'] * 100, 2)}, ") 
                    file.write(f"rouge2={round(eval_results['rouge2']* 100, 2)}, ") 
                    file.write(f"rougeL={round(eval_results['rougeL']* 100, 2)}, ") 
                    file.write(f"rougeLsum={round(eval_results['rougeLsum']* 100, 2)}, ") 
                    file.write(f"bleu={round(eval_results['bleu']* 100, 2)}, ") 
                    file.write(f"bertscore={round(eval_results['bertScore']* 100, 2)}\n") 

                    if('yes' in save):
                        df = pd.DataFrame({
                            'BLEU': [round(x * 100, 2) for x in eval_results['bleu_list']],
                            'Value': eval_results['predictions'],
                            'Label': eval_results['labels'],
                        }) #.sort_values(by='Label', ascending=True)

                        df.to_csv(f"{output_file}_inf", index=False)

                elapsed_time = (time.time() - start_time) / 60
                file.write(f"Time taken for evaluation: {elapsed_time:.2f} minutes\n")

                file.flush()
    finally:
        utils.cleanup()

def main():

    # Model names [use index for args.model parameter]
    model_names = [ 
        'ivabojic/whisper-small-sing2eng-transcribe',
        'ivabojic/whisper-medium-sing2eng-transcribe',
        'ivabojic/whisper-small-sing2eng-translate',
        'ivabojic/whisper-medium-sing2eng-translate',
        #'openai/whisper-small',
        #'openai/whisper-medium'
    ]

    valid_words = set(nltk.corpus.words.words())

    parser = argparse.ArgumentParser(description="Transcription/translation models evaluation")

    # Folder arguments with defaults
    parser.add_argument('--base_folder', default='output',
                        help="Base folder to store evaluation results.")
    parser.add_argument('--dataset_path', default='../small_dataset/test_transcribe.csv',
                        help="Path to transcription/translation test dataset.")
 
    # Evaluation configuration arguments
    parser.add_argument('--world_size', type=utils.restricted_world_size, default=1,
                        help="Number of GPUs for evaluation (default: 1)")
    parser.add_argument('--task', choices=['transcribe', 'translate'], default='transcribe',
                        help="Task type: transcribe or translate (default: transcribe)")
    parser.add_argument('--model', type=utils.restricted_models, default=0,
                        help="Model index: 0-3 (default: 0 (whisper_small))")
    parser.add_argument('--batch', type=utils.check_positive, default=4,
                        help="Batch size for evaluation (default: 4)")
    parser.add_argument('--max_length', type=int, default=448,
                        help="Max token length for evaluation (default: 448)")
    parser.add_argument('--language', default='English',
                        help="Language for evaluation (default: English)")
    parser.add_argument('--save', choices=['yes', 'no'], default='yes',
                        help="Saving results of evaluation (default: yes)")

    args = parser.parse_args()

    # Unpack arguments
    base_folder = args.base_folder
    dataset_path = args.dataset_path
    world_size = args.world_size
    task = args.task
    model_name = model_names[args.model]
    test_batch = args.batch
    MAX_LENGTH = args.max_length
    language = args.language
    save = args.save

    model_name_short = model_name.lower().split('/')[-1]
    print(f"Using model: {model_name_short}")
    
    # Determine output file based on task
    output_file = f"{base_folder}/evaluation/{task}_{model_name_short}.eval"

    mp.spawn(evaluate, args=(world_size, model_name, model_name_short, dataset_path, 
                output_file, MAX_LENGTH, language, task, test_batch, valid_words, save), 
                nprocs=world_size, join=True)

if __name__ == "__main__":

   main()
