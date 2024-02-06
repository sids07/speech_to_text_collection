from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor
from train_whisper.src.data_utils import load_data_into_datasets
from datasets import Audio
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

class DataPreprocessing:
    
    def __init__(self, model_name):
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
        self.tokenizer = WhisperTokenizer.from_pretrained(model_name, language="English", task="transcribe")
        self.processor = WhisperProcessor.from_pretrained(model_name, language="English", task="transcribe")
        
        self.data = load_data_into_datasets()
        self.sampling_rate = self.feature_extractor.sampling_rate
        
    def prepare_dataset(self,batch):
    # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array 
        batch["input_features"] = self.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # encode target text to label ids 
        batch["labels"] = self.tokenizer(batch["phrase"]).input_ids
        return batch
    
    def preprocess_data(self):
        
        data = self.data.cast_column("audio", Audio(sampling_rate=self.sampling_rate))
        
        data = data.map(self.prepare_dataset, remove_columns=data.column_names["train"], num_proc=8)
        return data
    
    def get_datacollator(self):
        return DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor)
    
    def get_processor(self):
        return self.processor
    
    def get_tokenizer(self):
        return self.tokenizer
