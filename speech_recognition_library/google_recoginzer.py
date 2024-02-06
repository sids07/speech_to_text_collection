import speech_recognition as sr
import whisper
import time
import soundfile
import numpy as np
import io
from faster_whisper import WhisperModel
from ctypes import *
from contextlib import contextmanager
import time
from whispercpp import Whisper

ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)

def py_error_handler(filename, line, function, err, fmt):
    pass

c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)

@contextmanager
def noalsaerr():
    asound = cdll.LoadLibrary('libasound.so')
    asound.snd_lib_error_set_handler(c_error_handler)
    yield
    asound.snd_lib_error_set_handler(None)



class STT:
    def __init__(self) -> None:
        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold=False
        self.whisper_model = whisper.load_model("base", device="cuda")
        # self.fast_whisper = WhisperModel("base", device="cuda")
        # self.whisper_cpp = Whisper.from_pretrained("base")
    
    def audio_transcribe(self,audio, stt_type):
        try:
            if stt_type == "google":
                print("Google Speech Recognition thinks you said " + self.recognizer.recognize_google(audio))
            
            if stt_type == "sphinx":
                print("Sphinx thinks you said " + self.recognizer.recognize_sphinx(audio))
            
            if stt_type == "whisper_offline":
                print("Whisper thinks you said " + self.recognizer.recognize_whisper(audio, language="english"))
            
            if stt_type=="whisper_api":
                print("Whisper API thinks you said" + self.recognizer.recognize_whisper_api(audio, api_key="YOUR_API_KEY"))

            if stt_type == "whisper_gpu":
                wav_bytes = audio.get_wav_data(convert_rate=16000)
                wav_stream = io.BytesIO(wav_bytes)
                audio_array, sampling_rate = soundfile.read(wav_stream)
                audio_array = audio_array.astype(np.float32)
                result = self.whisper_model.transcribe(
                audio_array,
                language="english"
            )
                print("Whisper GPU thinks you said" + result['text'])
            
            if stt_type == "azure":
                AZURE_SPEECH_KEY = "AZURE_SPEECH_KEY"  # Microsoft Speech API keys 32-character lowercase hexadecimal strings
                print(self.recognizer.recognize_azure(audio, key=AZURE_SPEECH_KEY, location="centralindia"))
            
            if stt_type == "fast_whisper":
                wav_bytes = audio.get_wav_data(convert_rate=16000)
                wav_stream = io.BytesIO(wav_bytes)
                audio_array, sampling_rate = soundfile.read(wav_stream)
                audio_array = audio_array.astype(np.float32)
                segments, info = self.fast_whisper.transcribe(audio_array, beam_size=5)
                print("Fast Whisper thinks:",segments)
            

        except sr.UnknownValueError:
            print("Recognition could not understand audio")
        
        except sr.RequestError as e:
            print("Could not request results from Speech Recognition service; {0}".format(e))


if __name__== "__main__":

    AUDIO_FILE = "/home/fm-pc-lt-275/Downloads/interview_0304/interview/interview-bot-2/src/audio/a1.wav"
    speech_to_text = STT()
    models = ["azure","google","sphinx","whisper_gpu","whisper_offline","whisper_api"]
    # models=["whisper_api","whisper_gpu","azure"]
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold=False

    with noalsaerr() as n, sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print('Say something')
        audio=recognizer.listen(source, timeout=2)
        print("Audio Received")

    for m in models:
        start = time.perf_counter()
        text = speech_to_text.audio_transcribe(audio=audio,
                                                    stt_type=m)
        print("Time Taken:", time.perf_counter()-start)
