from typing import Optional
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.httpsession import AwsCrtHttpSessionManager
from amazon_transcribe.model import StartStreamTranscriptionEventStream, StartStreamTranscriptionRequest, TranscriptEvent
from amazon_transcribe.serialize import TranscribeStreamingSerializer
from amazon_transcribe.signer import SigV4RequestSigner
from amazon_transcribe.request import Request

from pyaudio import Stream
import asyncio
import pyaudio
import aiofile
from amazon_transcribe.utils import apply_realtime_delay

SAMPLE_RATE = 48000
BYTES_PER_SAMPLE = 2
CHANNEL_NUMS = 1

# An example file can be found at tests/integration/assets/test.wav
AUDIO_PATH = "/home/fm-pc-lt-228/Desktop/upacare/speech-to-speech/speech_to_text/data/train/1249120_44142156_72079889.wav"
CHUNK_SIZE = 1024 * 8
REGION="us-east-1"

class MyEventHandler(TranscriptResultStreamHandler):
    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        # This handler can be implemented to handle transcriptions as needed.
        # Here's an example to get started.
        results = transcript_event.transcript.results
        for result in results:
            if result.is_partial is False:
                for alt in result.alternatives:
                    print(alt.transcript)
                
class StartMedicalStreamTranscriptionRequest(StartStreamTranscriptionRequest):
    def __init__(self, *args, **kwargs):
        audio_type = kwargs.pop("audio_type")
        specialty = kwargs.pop("specialty")

        super().__init__(*args, **kwargs)

        self.audio_type = audio_type
        self.specialty = specialty


class TranscribeMedicalStreamingSerializer(TranscribeStreamingSerializer):
    def __init__(self):
        super().__init__()

        self.request_uri = "/medical-stream-transcription"

    def serialize_start_stream_transcription_request(
        self, endpoint: str, request_shape: StartStreamTranscriptionRequest
    ) -> Request:
        request = super().serialize_start_stream_transcription_request(endpoint, request_shape)
        request.path = self.request_uri

        request.headers.update(
            super()._serialize_str_header(
                "specialty", request_shape.specialty
            )
        )
        
        request.headers.update(
            super()._serialize_str_header(
                "type", request_shape.audio_type
            )
        )

        return request

##


class TranscribeMedicalStreamingClient(TranscribeStreamingClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._serializer = TranscribeMedicalStreamingSerializer()

    async def start_stream_transcription(
        self,
        *,
        language_code: str,
        media_sample_rate_hz: int,
        media_encoding: str,
        audio_type: str,
        specialty: str,
        vocabulary_name: Optional[str] = None,
        session_id: Optional[str] = None,
        vocab_filter_method: Optional[str] = None,
        vocab_filter_name: Optional[str] = None,
        show_speaker_label: Optional[bool] = None,
        enable_channel_identification: Optional[bool] = None,
        number_of_channels: Optional[int] = None,
    ) -> StartStreamTranscriptionEventStream:
        transcribe_streaming_request = StartMedicalStreamTranscriptionRequest(
            language_code,
            media_sample_rate_hz,
            media_encoding,
            vocabulary_name,
            session_id,
            vocab_filter_method,
            vocab_filter_name,
            show_speaker_label,
            enable_channel_identification,
            number_of_channels,
            audio_type=audio_type,
            specialty=specialty,
        )
        endpoint = await self._endpoint_resolver.resolve(self.region)

        ## super
        request = self._serializer.serialize_start_stream_transcription_request(
            endpoint=endpoint, request_shape=transcribe_streaming_request,
        ).prepare()

        creds = await self._credential_resolver.get_credentials()
        signer = SigV4RequestSigner("transcribe", self.region)
        signed_request = signer.sign(request, creds)

        session = AwsCrtHttpSessionManager(self._eventloop)

        response = await session.make_request(
            signed_request.uri,
            method=signed_request.method,
            headers=signed_request.headers.as_list(),
            body=signed_request.body,
        )
        resolved_response = await response.resolve_response()

        status_code = resolved_response.status_code
        if status_code >= 400:
            # We need to close before we can consume the body or this will hang
            signed_request.body.close()
            body_bytes = await response.consume_body()
            raise self._response_parser.parse_exception(resolved_response, body_bytes)
        elif status_code != 200:
            raise RuntimeError("Unexpected status code encountered: %s" % status_code)

        parsed_response = self._response_parser.parse_start_stream_transcription_response(
            resolved_response,
            response,
        )

        # The audio stream is returned as output because it requires
        # the signature from the initial HTTP request to be useable
        audio_stream = self._create_audio_stream(signed_request)
        return StartStreamTranscriptionEventStream(audio_stream, parsed_response)
    
async def basic_transcribe():
    # Setup up our client with our chosen AWS region
    client = TranscribeMedicalStreamingClient(region="us-east-1")

    # Start transcription to generate our async stream
    stream = await client.start_stream_transcription(
        language_code="en-US",
        media_sample_rate_hz=SAMPLE_RATE,
        media_encoding="pcm",
        specialty="PRIMARYCARE",
        audio_type="CONVERSATION"
    )

    async def write_chunks():
        # NOTE: For pre-recorded files longer than 5 minutes, the sent audio
        # chunks should be rate limited to match the realtime bitrate of the
        # audio stream to avoid signing issues.
        async with aiofile.AIOFile(AUDIO_PATH, "rb") as afp:
            reader = aiofile.Reader(afp, chunk_size=CHUNK_SIZE)
            await apply_realtime_delay(
                stream, reader, BYTES_PER_SAMPLE, SAMPLE_RATE, CHANNEL_NUMS
            )
        await stream.input_stream.end_stream()
        
    # Instantiate our handler and start processing events
    handler = MyEventHandler(stream.output_stream)
    await asyncio.gather(write_chunks(), handler.handle_events())
    
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(basic_transcribe())
    loop.close()
    # SAMPLE_RATE = 16000
    # FRAMES_PER_BUFFER = 4096
    # p = pyaudio.PyAudio()
    # audio_stream = p.open(
    #     frames_per_buffer=FRAMES_PER_BUFFER,
    #     # input_device_index=1,
    #     rate=SAMPLE_RATE,
    #     format=pyaudio.paInt16,
    #     channels=1,
    #     input=True,
    # )

    # asyncio.run(
    #     basic_transcribe(
    #         audio_stream=audio_stream,
    #         sample_rate=SAMPLE_RATE,
    #         chunk_size=FRAMES_PER_BUFFER,
    #     )
    # )