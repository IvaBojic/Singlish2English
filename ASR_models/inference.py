import torch, torchaudio, argparse
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Model names [use index for args.model parameter]
model_names = [ 
    'ivabojic/whisper-small-sing2eng-transcribe',
    'ivabojic/whisper-medium-sing2eng-transcribe',
    'ivabojic/whisper-small-sing2eng-translate',
    'ivabojic/whisper-medium-sing2eng-translate',
    #'openai/whisper-small',
    #'openai/whisper-medium'
]

parser = argparse.ArgumentParser(description="Transcription/translation models inference")

parser.add_argument('--audio_path', default='../small_dataset/audios/00862042_713.wav',
                        help="Path to audio file.")
parser.add_argument('--model', default=0,
                    help="Model index: 0-3 (default: 0 (whisper_small))")
parser.add_argument('--language', default='English',
                        help="Language for evaluation (default: English)")
parser.add_argument('--task', choices=['transcribe', 'translate'], default='transcribe',
                        help="Task type: transcribe or translate (default: transcribe)")

args = parser.parse_args()

# Unpack arguments
audio_path= args.audio_path
model_name = model_names[args.model]
language = args.language
task = args.task

model = WhisperForConditionalGeneration.from_pretrained(model_name) 
processor = WhisperProcessor.from_pretrained(model_name, language=language, task=task) 

audio, sampling_rate = torchaudio.load(audio_path)

 # Resample the audio to 16000 Hz
if sampling_rate != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
    audio = resampler(audio)
    sampling_rate = 16000 

# Convert audio to numpy array and extract features
audio = audio.squeeze().numpy()

# Preprocess
inputs = processor(audio=audio, sampling_rate=sampling_rate, return_tensors="pt")

# Generate
with torch.no_grad():
    predicted_ids = model.generate(inputs.input_features)

# Decode
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print(transcription)
