from transformers import pipeline
from train_whisper.config.config import TRAINING_ARGS
import time

class Inference:
    
    def __init__(
        self, 
        model_name = TRAINING_ARGS["hub_model_id"],
        device="cuda"
        ):
        
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name, 
            device=device)
    
    def get_transcription(self,audio):
        start = time.perf_counter()
        pipeline_result =  self.pipe(audio)
        transcription = pipeline_result["text"]
        print(transcription)
        print("Time taken: ", time.perf_counter()-start)
        return transcription