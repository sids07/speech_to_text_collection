from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent

from pyaudio import Stream
import asyncio
import pyaudio

"""
Here's an example of a custom event handler you can extend to
process the returned transcription results as needed. This
handler will simply print the text out to your interpreter.
"""


BYTES_PER_SAMPLE = 2
CHANNEL_NUMS = 1

# An example file can be found at tests/integration/assets/test.wav
REGION = "us-east-1"


class MyEventHandler(TranscriptResultStreamHandler):
    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        # This handler can be implemented to handle transcriptions as needed.
        # Here's an example to get started.
        results = transcript_event.transcript.results
        for result in results:
            for alt in result.alternatives:
                print(alt.transcript)


async def basic_transcribe(audio_stream: Stream, sample_rate: int, chunk_size: int):
    # Setup up our client with our chosen AWS region
    client = TranscribeStreamingClient(region=REGION)

    # Start transcription to generate our async stream
    stream = await client.start_stream_transcription(
        language_code="en-US",
        media_sample_rate_hz=sample_rate,
        media_encoding="pcm",
    )

    async def write_chunks():
        # NOTE: For pre-recorded files longer than 5 minutes, the sent audio
        # chunks should be rate limited to match the realtime bitrate of the
        # audio stream to avoid signing issues.

        # what we do now is basically 'for chunk in audio_stream'
        # but we need to do it asynchronously:
        while True:
            chunk = audio_stream.read(chunk_size)
            if not chunk:
                break
            await stream.input_stream.send_audio_event(audio_chunk=chunk)
        await stream.input_stream.end_stream()

    # Instantiate our handler and start processing events
    handler = MyEventHandler(stream.output_stream)
    await asyncio.gather(write_chunks(), handler.handle_events())
    

if __name__ == "__main__":
    SAMPLE_RATE = 16000
    FRAMES_PER_BUFFER = 4096
    p = pyaudio.PyAudio()
    audio_stream = p.open(
        frames_per_buffer=FRAMES_PER_BUFFER,
        # input_device_index=1,
        rate=SAMPLE_RATE,
        format=pyaudio.paInt16,
        channels=1,
        input=True,
    )

    asyncio.run(
        basic_transcribe(
            audio_stream=audio_stream,
            sample_rate=SAMPLE_RATE,
            chunk_size=FRAMES_PER_BUFFER,
        )
    )