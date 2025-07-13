import torchaudio
from torch.utils.data import Dataset

class AudioTextDataset(Dataset):
    
    def __init__(self, dataset, processor, tokenizer, max_input_length, model_name):
        self.dataset = dataset
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.model_name = model_name      

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get the sample from the Dataset
        batch = self.dataset[idx]
        audio_path = batch["Path"]
        transcription = batch["Value"]

        # Load audio file
        audio, sampling_rate = torchaudio.load(audio_path)

        # Resample the audio to 16000 Hz
        if sampling_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
            audio = resampler(audio)

        # Convert audio to numpy array and extract features
        audio = audio.squeeze().numpy()

        model_type = self.model_name.lower()
        if(model_type.count("/") > 1):     
            model_type = model_type.split("/")[-2]

        if("t5" in model_type):
            inputs = self.processor(audio=audio, sampling_rate=16000, return_tensors="pt", padding="max_length",
                                            truncation=True, max_length=480000)
            input_features = inputs['input_values'][0]
        elif("seamless" in model_type):
            inputs = self.processor(raw_speech=audio, sampling_rate=16000, padding="max_length", 
                                        truncation=True, max_length=self.max_input_length * 2, return_tensors="pt")
            input_features = inputs.input_features[0]
        elif("whisper" in model_type):
            inputs = self.processor(raw_speech=audio, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features[0]

        # Tokenize the transcription
        encoding = self.tokenizer(text=transcription, padding="max_length", max_length=self.max_input_length, 
                                  truncation=True, return_tensors="pt")
        labels = encoding.input_ids[0]

        return {
            "input_features": input_features,  # type: ignore
            "labels": labels
        }