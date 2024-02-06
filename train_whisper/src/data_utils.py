import os
import pandas as pd
from datasets import DatasetDict, Dataset
from speech_to_text.train_whisper.config.config import DATA_OVERVIEW

def add_audio_column(category, file_name):
    
    if category == "train":
        return os.path.join("data/test",file_name)
    elif category == "validate":
        return os.path.join("data/validate",file_name)
    elif category == "test":
        return os.path.join("data/train",file_name)
    

def create_data():
    
    df = pd.read_csv(DATA_OVERVIEW)
    
    df["audio"] = df.apply(lambda x: add_audio_column(x.category, x.file_name), axis=1)
    
    train_df = df[df["category"]=="train"].reset_index(drop=True)
    valid_df = df[df["category"]=="validate"].reset_index(drop=True)
    test_df = df[df["category"]=="test"].reset_index(drop=True)
    
    train_df = train_df[["audio","phrase"]]
    valid_df = valid_df[["audio","phrase"]]
    test_df = test_df[["audio","phrase"]]
    
    return train_df, valid_df, test_df

def load_data_into_datasets():
    
    train_df, valid_df, test_df = create_data()
    
    data = DatasetDict()
    data["train"] = Dataset.from_pandas(train_df)
    data["test"] = Dataset.from_pandas(valid_df)
    
    return data