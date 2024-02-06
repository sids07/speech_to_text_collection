import os
import azure.cognitiveservices.speech as speechsdk
import threading
import time

def recognize_from_microphone():
    # This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
    # speechsdk.PropertyCollection().set_property(speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, "100000000")
    # speechsdk.PropertyCollection().set_property(speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, "100000000")
    

    speech_config = speechsdk.SpeechConfig(subscription=os.environ.get("SPEECH_KEY"), region=os.environ.get("SPEECH_REGION"))
    # speech_config = speechsdk.SpeechConfig(subscription="AZURE_API_KEY", region="centralindia")

    speech_config.speech_recognition_language="en-PH"
    # speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, "100000000")
    speech_config.set_property(speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, "2000")
    # speech_config.enable_dictation()
    speech_config.set_profanity(speechsdk.ProfanityOption.Raw)
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    # print(speechsdk.PropertyId("SpeechServiceConnection_EndSilenceTimeoutMs"))
    # print(speechsdk.PropertyCollection.get_property('SpeechServiceConnection_EndSilenceTimeoutMs','0'))
    phrase_list_grammar = speechsdk.PhraseListGrammar.from_recognizer(speech_recognizer)
    phrase_list_grammar.addPhrase("Dhan bahadur")
    phrase_list_grammar.addPhrase("K gardai chau")
    print("CC",phrase_list_grammar)
    def recognizing_cb(evt: speechsdk.SpeechRecognitionEventArgs):
        print('RECOGNIZING: {}'.format(evt.result.text))
        print("Offset in Ticks: {}".format(evt.result.offset))
        print("Duration in Ticks: {}".format(evt.result.duration))
        global ts
        ts = time.perf_counter()
        print("TS", ts)
    # Connect callbacks to the events fired by the speech recognizer
    def recognized_cb(evt: speechsdk.SpeechRecognitionEventArgs):
        print('RECOGNIZED: {}'.format(evt))

    speech_recognizer.recognizing.connect(recognizing_cb)
    speech_recognizer.recognized.connect(recognized_cb)


    print("Speak into your microphone.")
    ss = time.perf_counter()
    speech_recognition_result = speech_recognizer.recognize_once_async().get()
    print("Total time for timeout:", time.perf_counter()-ss)
    if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(speech_recognition_result.text))
        print("Offset in Ticks: {}".format(speech_recognition_result.offset))
        print("Duration in Ticks: {}".format(speech_recognition_result.duration))
        print("FInal TS", ts)
        print("Time Taken", time.perf_counter()-ts)
    elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(speech_recognition_result.no_match_details))
    elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_recognition_result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))
            print("Did you set the speech resource key and region values?")

def speech_language_detection_once_from_mic():
    """performs one-shot speech language detection from the default microphone"""
    # <SpeechLanguageDetectionWithMicrophone>
    # Creates an AutoDetectSourceLanguageConfig, which defines a number of possible spoken languages
    auto_detect_source_language_config = \
        speechsdk.languageconfig.AutoDetectSourceLanguageConfig(languages=["en-SG","en-US","en-PH"])

    # Creates a SpeechConfig from your speech key and region
    speech_config = speechsdk.SpeechConfig(subscription=os.environ.get("SPEECH_KEY"), region=os.environ.get("SPEECH_REGION"))

    # Creates a source language recognizer using microphone as audio input.
    # The default language is "en-us".
    speech_language_detection = speechsdk.SourceLanguageRecognizer(
        speech_config=speech_config, auto_detect_source_language_config=auto_detect_source_language_config)

    print("Say something in English or German...")

    # Starts speech language detection, and returns after a single utterance is recognized. The end of a
    # single utterance is determined by listening for silence at the end or until a maximum of 15
    # seconds of audio is processed. It returns the recognition text as result.
    # Note: Since recognize_once() returns only a single utterance, it is suitable only for single
    # shot recognition like command or query.
    # For long-running multi-utterance recognition, use start_continuous_recognition() instead.
    result = speech_language_detection.recognize_once()

    # Check the result
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        detected_src_lang = result.properties[
            speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult]
        print("Detected Language: {}".format(detected_src_lang))
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized")
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))

def speech_recognize_continuous_async_from_microphone():
    """performs continuous speech recognition asynchronously with input from microphone"""
    speech_config = speechsdk.SpeechConfig(subscription=os.environ.get("SPEECH_KEY"), region=os.environ.get("SPEECH_REGION"))
    speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, "500")
    speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "500")
    speech_config.set_property(speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, "500")

    # The default language is "en-us".
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)

    done = False

    def recognizing_cb(evt: speechsdk.SpeechRecognitionEventArgs):
        print('RECOGNIZING: {}'.format(evt))

    def recognized_cb(evt: speechsdk.SpeechRecognitionEventArgs):
        print('RECOGNIZED: {}'.format(evt))

    def stop_cb(evt: speechsdk.SessionEventArgs):
        """callback that signals to stop continuous recognition"""
        print('CLOSING on {}'.format(evt))
        nonlocal done
        done = True

    # Connect callbacks to the events fired by the speech recognizer
    speech_recognizer.recognizing.connect(recognizing_cb)
    speech_recognizer.recognized.connect(recognized_cb)
    speech_recognizer.session_stopped.connect(stop_cb)
    speech_recognizer.canceled.connect(stop_cb)

    # Perform recognition. `start_continuous_recognition_async asynchronously initiates continuous recognition operation,
    # Other tasks can be performed on this thread while recognition starts...
    # wait on result_future.get() to know when initialization is done.
    # Call stop_continuous_recognition_async() to stop recognition.
    result_future = speech_recognizer.start_continuous_recognition_async()

    result_future.get()  # wait for voidfuture, so we know engine initialization is done.
    print('Continuous Recognition is now running, say something.')

    while not done:
        # No real sample parallel work to do on this thread, so just wait for user to type stop.
        # Can't exit function or speech_recognizer will go out of scope and be destroyed while running.
        print('type "stop" then enter when done')
        stop = input()
        if (stop.lower() == "stop"):
            print('Stopping async recognition.')
            speech_recognizer.stop_continuous_recognition_async()
            break
    
    speech_recognizer.stop_continuous_recognition_async()
    # while not done:
    #     if
    #     speech_recognizer.stop_continuous_recognition_async()


    print("recognition stopped, main thread can exit now.")


if __name__ == "__main__":
    recognize_from_microphone()
