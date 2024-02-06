#!/usr/bin/env python
# coding: utf-8

# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See https://aka.ms/csspeech/license for the full license information.

from datetime import datetime
from timeit import default_timer as timer
import threading
import os

try:
    import azure.cognitiveservices.speech as speechsdk
except ImportError:
    print("""
    Importing the Speech SDK for Python failed.
    Refer to
    https://docs.microsoft.com/azure/cognitive-services/speech-service/quickstart-python for
    installation instructions.
    """)
    import sys
    sys.exit(1)

os.environ["SPEECH_KEY"] = "AZURE_SPEECH_KEY"
os.environ["SPEECH_REGION"] = "centralindia"


def print_ts(str):
    # print str with a timestamp prefix
    dt = datetime.now()
    print(f"[{dt:%H:%M:%S}.{f'{dt:%f}'[:3]}] " + str)


def recognize_speech_from_microphone():
    start = timer()
    print_ts('ENTER recognize_speech_from_microphone')

    # create speech config
    speech_config = speechsdk.SpeechConfig(subscription=os.environ.get("SPEECH_KEY"), region=os.environ.get("SPEECH_REGION"))
    #speech_config.set_property(speechsdk.PropertyId.Speech_LogFilename, "sdk.log")
    speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "4500")
    speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, "1500")
    # speech_config.set_property(speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, "2000")
    speech_config.output_format = speechsdk.OutputFormat.Detaileds

    # (optional) create audio config
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)

    # instantiate the speech recognizer
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    recognition_done = threading.Event()

    # callbacks for specific events
    def recognized_cb(evt):
        try:
            result = evt.result
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                print_ts('RECOGNIZED: {}'.format(result.text))
                if not result.text:  # workaround for possible service issues in effect
                    print_ts('Closing because of presumed silence timeout')
                    recognition_done.set()
            elif result.reason == speechsdk.ResultReason.NoMatch:
                print_ts('NOMATCH: {}'.format(result.no_match_details.reason))
                if result.no_match_details.reason == speechsdk.NoMatchReason.InitialSilenceTimeout:
                    print_ts('Closing because of InitialSilenceTimeout')
                    recognition_done.set()
                if result.no_match_details.reason == speechsdk.NoMatchReason.EndSilenceTimeout:
                    print_ts('Closing because of EndSilenceTimeout')
                    recognition_done.set()
        except Exception as e:
            print(e)

    def canceled_cb(evt):
        try:
            result = evt.result
            if result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = result.cancellation_details
                print_ts('CANCELED: {}'.format(cancellation_details.reason))
                if cancellation_details.reason == speechsdk.CancellationReason.Error:
                    print('Error details: {}'.format(cancellation_details.error_details))
                    recognition_done.set()
        except Exception as e:
            print(e)

    # connect callbacks to the events fired by the speech recognizer
    speech_recognizer.recognizing.connect(lambda evt: print_ts('Recognizing: {}'.format(evt.result.text)))
    speech_recognizer.recognized.connect(recognized_cb)
    speech_recognizer.session_started.connect(lambda evt: print_ts('Session started: {}'.format(evt)))
    speech_recognizer.session_stopped.connect(lambda evt: print_ts('Session stopped: {}'.format(evt)))
    speech_recognizer.canceled.connect(canceled_cb)

    # start continuous speech recognition
    speech_recognizer.start_continuous_recognition_async().get()
    # wait until timeout or canceled
    recognition_done.wait()
    # stop continuous speech recognition
    speech_recognizer.stop_continuous_recognition_async().get()

    print_ts('EXIT recognize_speech_from_microphone')
    end = timer()
    print("%.3f s" % (end - start))


if __name__ == '__main__':
    recognize_speech_from_microphone()
