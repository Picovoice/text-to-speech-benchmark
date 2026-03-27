import asyncio
import base64
import json
import os
import subprocess
import threading
import time
from contextlib import closing
from dataclasses import dataclass
from enum import Enum
from queue import Queue
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Generator,
    Literal,
    Optional
)

import numpy as np
import pvorca
import requests
import torch
import websockets
from openai import OpenAI
from pvorca import OrcaActivationLimitError

from audio import (
    AudioEncodings,
    AudioSink
)

from ._measurement import (
    CoreTimeMeasure,
    Timer,
    include_measurement
)

INT16_SCALE = 32767


class Synthesizers(Enum):
    AZURE_TTS = "azure_tts"
    AMAZON_POLLY = "amazon_polly"
    ELEVENLABS = "elevenlabs"
    ELEVENLABS_WEBSOCKET = "elevenlabs_websocket"
    OPENAI_TTS = "openai_tts"
    PICOVOICE_ORCA = "picovoice_orca"
    KOKORO_TTS = "kokoro_tts"
    CHATTERBOX_TTS_TURBO = "chatterbox_tts_turbo"
    KITTEN_TTS = "kitten_tts"
    POCKET_TTS = "pocket_tts"
    NEU_TTS_NANO_Q4_GGUF = "neu_tts_nano_q4_gguf"
    PIPER_TTS = "piper_tts"
    SOPRANO_TTS = "soprano_tts"
    SUPERTONIC_TTS_2 = "supertonic_tts_2"
    ESPEAK_NG = "espeak_ng"


class Synthesizer:
    def __init__(
            self,
            sample_rate: int,
            audio_encoding: AudioEncodings,
            timer: Timer,
            text_streamable: bool = False,
    ) -> None:
        self.sample_rate = sample_rate
        self.text_streamable = text_streamable

        self._timer = timer
        self._audio_sink = AudioSink(sample_rate=self.sample_rate, encoding=audio_encoding)

    def synthesize(self, text_stream: Generator[str, None, None]) -> None:
        raise NotImplementedError(
            f"Method `synthesize` must be implemented in a subclass of {self.__class__.__name__}")

    async def synthesize_async(self, text_stream: AsyncGenerator[str, None]) -> None:
        raise NotImplementedError(
            f"Method `synthesize` must be implemented in a subclass of {self.__class__.__name__}")

    def terminate(self) -> None:
        pass

    def _read_text_stream(self, text_stream: Generator[str, None, None]) -> str:
        text = ""
        for token in text_stream:
            self._timer.maybe_log_time_first_llm_token()
            text += token
            self._timer.increment_num_tokens()

        self._timer.log_time_last_llm_token()

        return text

    def save_and_reset_last_audio(self, path: str) -> None:
        self._audio_sink.save(path)
        self._audio_sink.reset()

    @property
    def is_async(self) -> bool:
        return False

    @classmethod
    def create(cls, engine: Synthesizers, **kwargs: Any) -> 'Synthesizer':
        subclasses = {
            Synthesizers.AMAZON_POLLY: AmazonSynthesizer,
            Synthesizers.AZURE_TTS: AzureSynthesizer,
            Synthesizers.ELEVENLABS: ElevenLabsSynthesizer,
            Synthesizers.ELEVENLABS_WEBSOCKET: ElevenLabsWebSocketSynthesizer,
            Synthesizers.OPENAI_TTS: OpenAISynthesizer,
            Synthesizers.PICOVOICE_ORCA: PicovoiceOrcaSynthesizer,
            Synthesizers.KOKORO_TTS: KokoroSynthesizer,
            Synthesizers.CHATTERBOX_TTS_TURBO: ChatterboxTurboSynthesizer,
            Synthesizers.KITTEN_TTS: KittenSynthesizer,
            Synthesizers.POCKET_TTS: PocketSynthesizer,
            Synthesizers.NEU_TTS_NANO_Q4_GGUF: NeuTTSNanoSynthesizer,
            Synthesizers.PIPER_TTS: PiperSynthesizer,
            Synthesizers.SOPRANO_TTS: SopranoSynthesizer,
            Synthesizers.SUPERTONIC_TTS_2: Supertonic2Synthesizer,
            Synthesizers.ESPEAK_NG: EspeakNGSynthesizer,
        }

        if engine not in subclasses:
            raise NotImplementedError(f"Cannot create {cls.__name__} of type `{engine.value}`")

        return subclasses[engine](**kwargs)

    def __str__(self) -> str:
        raise NotImplementedError()


class ElevenLabsSynthesizer(Synthesizer):
    NAME = "ElevenLabs"

    SAMPLE_RATE = 22050
    AUDIO_ENCODING = AudioEncodings.BYTES
    CHUNK_SIZE = 10 * 1024

    VOICE_ID = "EXAVITQu4vr4xnSDxMaL"
    MODEL_ID = "eleven_turbo_v2"
    URL_TEMPLATE = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream?" \
        "model_id=eleven_turbo_v2_5&output_format=pcm_22050"

    def __init__(
            self,
            api_key: str,
            save_audio: bool = True,
            **kwargs: Any,
    ) -> None:
        super().__init__(sample_rate=self.SAMPLE_RATE, audio_encoding=self.AUDIO_ENCODING, **kwargs)

        self._save_audio = save_audio

        self._headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json"}
        self._url = self.URL_TEMPLATE.format(voice_id=self.VOICE_ID)

    def _build_payload(self, text: str) -> Dict[str, Any]:
        return {
            "text": text,
            "model_id": self.MODEL_ID,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.8,
                "use_speaker_boost": False,
            },
            "seed": 77777,
        }

    def synthesize(self, text_stream: Generator[str, None, None]) -> None:
        payload = self._build_payload(text=self._read_text_stream(text_stream))

        self._timer.log_time_first_synthesis_request()

        response = requests.request(
            "POST",
            self._url,
            json=payload,
            headers=self._headers,
            params={"output_format": "pcm_22500"}
        )

        for chunk in response.iter_content(chunk_size=self.CHUNK_SIZE):
            self._timer.maybe_log_time_first_audio()

            if self._save_audio:
                self._audio_sink.add(data=chunk)

        self._timer.log_time_last_audio()

    def __str__(self) -> str:
        return f"{self.NAME}"


class ElevenLabsWebSocketSynthesizer(Synthesizer):
    NAME = "ElevenLabs WebSocket"

    SAMPLE_RATE = 22050
    AUDIO_ENCODING = AudioEncodings.BYTES

    VOICE_ID = "EXAVITQu4vr4xnSDxMaL"
    URI = \
        "wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input?" \
        "model_id=eleven_turbo_v2_5&output_format=pcm_22050"

    def __init__(
            self,
            api_key: str,
            save_audio: bool = True,
            **kwargs: Any,
    ) -> None:
        super().__init__(sample_rate=self.SAMPLE_RATE, audio_encoding=self.AUDIO_ENCODING, **kwargs)

        self._api_key = api_key
        self._uri = self.URI.format(voice_id=self.VOICE_ID)

        self._save_audio = save_audio

    async def _text_chunker(self, text_stream: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
        splitters = (".", ",", "?", "!", ";", ":", "—", "-", "(", ")", "[", "]", "}", " ")
        buffer = ""

        async for text in text_stream:
            if text is None:
                continue
            self._timer.maybe_log_time_first_llm_token()
            if buffer.endswith(splitters):
                yield buffer + " "
                buffer = text
            elif text.startswith(splitters):
                yield buffer + text[0] + " "
                buffer = text[1:]
            else:
                buffer += text
            self._timer.increment_num_tokens()

        if buffer:
            yield buffer + " "

    async def synthesize_async(self, text_stream: AsyncGenerator[str, None]) -> None:
        async with websockets.connect(self._uri) as websocket:
            await websocket.send(json.dumps({
                "text": " ",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.8},
                "xi_api_key": self._api_key,
            }))

            async def consume_audio() -> None:
                while True:
                    try:
                        message = await websocket.recv()
                        data = json.loads(message)
                        if data.get("audio"):
                            self._timer.maybe_log_time_first_audio()
                            if self._save_audio:
                                self._audio_sink.add(data=base64.b64decode(data["audio"]))
                        elif data.get('isFinal'):
                            break
                    except websockets.exceptions.ConnectionClosed:
                        break

            task_consume_audio = asyncio.create_task(consume_audio())

            async for text in self._text_chunker(text_stream=text_stream):
                self._timer.maybe_log_time_first_synthesis_request()
                await websocket.send(json.dumps({"text": text, "try_trigger_generation": True}))

            await websocket.send(json.dumps({"text": ""}))
            self._timer.log_time_last_llm_token()

            await task_consume_audio

        self._timer.log_time_last_audio()

    @property
    def is_async(self) -> bool:
        return True

    def __str__(self) -> str:
        return f"{self.NAME}"


class AzureSynthesizer(Synthesizer):
    NAME = "Azure TTS"

    SAMPLE_RATE = 24000
    CHUNK_SIZE = 10 * 1024
    AUDIO_ENCODING = AudioEncodings.BYTES
    VOICE_NAME = "en-CA-ClaraNeural"

    def __init__(
            self,
            speech_key: str,
            speech_region: str,
            save_audio: bool = True,
            **kwargs: Any
    ) -> None:
        # noinspection PyPackageRequirements
        import azure.cognitiveservices.speech as speechsdk
        super().__init__(sample_rate=self.SAMPLE_RATE, audio_encoding=self.AUDIO_ENCODING, **kwargs)

        self._save_audio = save_audio

        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
        speech_config.speech_synthesis_voice_name = self.VOICE_NAME
        speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Raw24Khz16BitMonoPcm)

        self._synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)

    def synthesize(self, text_stream: Generator[str, None, None]) -> None:
        # noinspection PyPackageRequirements
        import azure.cognitiveservices.speech as speechsdk

        text = self._read_text_stream(text_stream)

        self._timer.log_time_first_synthesis_request()

        result = self._synthesizer.start_speaking_text_async(text).get()
        buffer = bytes(self.CHUNK_SIZE)
        stream = speechsdk.AudioDataStream(result)

        num_reads = stream.read_data(buffer)
        while num_reads > 0:
            self._timer.maybe_log_time_first_audio()
            if self._save_audio:
                self._audio_sink.add(data=buffer)
            buffer = bytes(self.CHUNK_SIZE)
            num_reads = stream.read_data(buffer)

        self._timer.log_time_last_audio()

    def __str__(self) -> str:
        return f"{self.NAME}"


class AmazonSynthesizer(Synthesizer):
    NAME = "Amazon Polly"

    SAMPLE_RATE = 22050
    CHUNK_SIZE = 10 * 1024
    VOICE = "Joanna"

    def __init__(
            self,
            aws_profile_name: str,
            save_audio: bool = True,
            **kwargs: Any,
    ) -> None:
        super().__init__(
            sample_rate=self.SAMPLE_RATE,
            audio_encoding=AudioEncodings.FILE_BUFFER,
            **kwargs)

        self._save_audio = save_audio

        from boto3 import Session
        session = Session(profile_name=aws_profile_name)
        self._client = session.client("polly")

    def synthesize(self, text_stream: Generator[str, None, None]) -> None:
        text = self._read_text_stream(text_stream)

        self._timer.maybe_log_time_first_synthesis_request()

        response = self._client.synthesize_speech(
            Text=text,
            SampleRate=str(self.sample_rate),
            OutputFormat="mp3",
            VoiceId="Joanna")

        if "AudioStream" in response:
            with closing(response["AudioStream"]) as stream:
                data = stream.read(self.CHUNK_SIZE)
                while len(data) > 0:
                    self._timer.maybe_log_time_first_audio()
                    if self._save_audio:
                        self._audio_sink.add(data=data)
                    data = stream.read(self.CHUNK_SIZE)
        else:
            raise ValueError(f"Failed to synthesize text: `{text}`")

        self._timer.log_time_last_audio()

    def __str__(self) -> str:
        return f"{self.NAME}"


class OpenAISynthesizer(Synthesizer):
    NAME = "OpenAI TTS"

    SAMPLE_RATE = 24000
    AUDIO_ENCODING = AudioEncodings.BYTES
    CHUNK_SIZE = 10 * 1024
    DEFAULT_MODEL_NAME = "tts-1"
    DEFAULT_VOICE_NAME = "shimmer"

    def __init__(
            self,
            api_key: str,
            model_name: str = DEFAULT_MODEL_NAME,
            voice_name: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"] = DEFAULT_VOICE_NAME,
            save_audio: bool = True,
            **kwargs: Any
    ) -> None:
        super().__init__(
            sample_rate=self.SAMPLE_RATE,
            audio_encoding=self.AUDIO_ENCODING,
            **kwargs)

        self._model_name = model_name
        self._voice_name = voice_name
        self._save_audio = save_audio

        self._client = OpenAI(api_key=api_key)

    def synthesize(self, text_stream: Generator[str, None, None]) -> None:
        text = self._read_text_stream(text_stream)

        self._timer.maybe_log_time_first_synthesis_request()

        with self._client.audio.speech.with_streaming_response.create(
            model=self._model_name,
            voice=self._voice_name,
            response_format="pcm",
            input=text
        ) as response:
            for data in response.iter_bytes(chunk_size=self.CHUNK_SIZE):
                self._timer.maybe_log_time_first_audio()
                if self._save_audio:
                    self._audio_sink.add(data=data)
            self._timer.log_time_last_audio()

    def __str__(self) -> str:
        return f"{self.NAME}"


class PicovoiceOrcaSynthesizer(Synthesizer):
    NAME = "Picovoice Orca"
    AUDIO_ENCODING = AudioEncodings.INT16

    @dataclass
    class OrcaTextInput:
        text: str
        flush: bool

    def __init__(
            self,
            timer: Timer,
            access_key: str,
            model_path: Optional[str] = None,
            device: Optional[str] = None,
            library_path: Optional[str] = None,
            save_audio: bool = True,
    ) -> None:
        self._orca = pvorca.create(
            access_key=access_key,
            model_path=model_path,
            device=device,
            library_path=library_path)
        super().__init__(
            sample_rate=self._orca.sample_rate,
            timer=timer,
            text_streamable=True,
            audio_encoding=self.AUDIO_ENCODING)

        self._orca_stream = self._orca.stream_open()

        self._queue: Queue[Optional[PicovoiceOrcaSynthesizer.OrcaTextInput]] = Queue()

        self._num_tokens = 0

        self._save_audio = save_audio

        self._thread = None
        self._start_thread()

    def _start_thread(self) -> None:
        self._thread = threading.Thread(target=self._run)
        self._thread.start()

    def _close_thread_blocking(self):
        self._queue.put_nowait(None)
        self._thread.join()

    def _run(self) -> None:
        while True:
            orca_input = self._queue.get()
            if orca_input is None:
                self._timer.log_time_last_audio()
                break

            time_before_proc = time.perf_counter()
            try:
                if not orca_input.flush:
                    pcm = self._orca_stream.synthesize(orca_input.text)
                else:
                    pcm = self._orca_stream.flush()
            except OrcaActivationLimitError:
                raise ValueError("Orca activation limit reached.")

            if pcm is not None:
                self._timer.maybe_set_time_first_synthesis_request(seconds=time_before_proc)
                self._timer.maybe_log_time_first_audio()
                pcm_seconds = len(pcm) / self._orca.sample_rate
                self._timer.accumulate_audio_seconds(pcm_seconds)
                if self._save_audio:
                    self._audio_sink.add(data=pcm)

    def synthesize(
            self,
            text_stream: Generator[str, None, None],
    ) -> None:
        with CoreTimeMeasure() as core_time_measure:
            for token in text_stream:
                self._timer.maybe_log_time_first_llm_token()
                with include_measurement(core_time_measure):
                    self._synthesize(token)
                self._timer.increment_num_tokens()

            self._timer.log_time_last_llm_token()

            with include_measurement(core_time_measure):
                self._flush()

            self._timer.core_time = core_time_measure.accum_time

    def _synthesize(self, text: str) -> None:
        self._queue.put_nowait(self.OrcaTextInput(text=text, flush=False))

    def _flush(self) -> None:
        self._queue.put_nowait(self.OrcaTextInput(text="", flush=True))
        self._close_thread_blocking()
        self._start_thread()

    def terminate(self):
        self._close_thread_blocking()
        self._orca_stream.close()
        self._orca.delete()

    def __str__(self) -> str:
        return f"{self.NAME}"


class KokoroSynthesizer(Synthesizer):
    NAME = "Kokoro TTS"
    SAMPLE_RATE = 24000
    AUDIO_ENCODING = AudioEncodings.INT16
    LANGUAGE_CODE = "a"
    VOICE_ID = "af_heart"
    SPEED = 1
    DEVICE = "cpu"

    def __init__(
            self,
            save_audio: bool = True,
            **kwargs: Any,
    ) -> None:
        super().__init__(
            sample_rate=self.SAMPLE_RATE,
            audio_encoding=self.AUDIO_ENCODING,
            **kwargs,
        )

        self._save_audio = save_audio

        from kokoro import KPipeline

        self._pipeline = KPipeline(
            lang_code=self.LANGUAGE_CODE,
            device=self.DEVICE,
        )

    def synthesize(
            self,
            text_stream: Generator[
                str,
                None,
                None,
            ],
    ):
        text = self._read_text_stream(text_stream)

        with CoreTimeMeasure() as core_time_measure:
            with include_measurement(core_time_measure):
                self._timer.maybe_log_time_first_synthesis_request()

                generator = self._pipeline(
                    text,
                    voice=self.VOICE_ID,
                    speed=self.SPEED,
                    split_pattern=r'\n+',
                )

                for gs, ps, chunk in generator:
                    self._timer.maybe_log_time_first_audio()

                    chunk = torch.clamp(
                        chunk,
                        -1,
                        1,
                    ) * INT16_SCALE
                    chunk = chunk.to(torch.int16).numpy()

                    chunk_seconds = len(chunk) / self.SAMPLE_RATE
                    self._timer.accumulate_audio_seconds(chunk_seconds)
                    if self._save_audio:
                        self._audio_sink.add(data=chunk)

                self._timer.log_time_last_audio()

            self._timer.core_time = core_time_measure.accum_time

    def __str__(self) -> str:
        return f"{self.NAME}"


class ChatterboxTurboSynthesizer(Synthesizer):
    NAME = "Chatterbox TTS Turbo"
    SAMPLE_RATE = 24000
    AUDIO_ENCODING = AudioEncodings.INT16
    DEVICE = "cpu"

    def __init__(
            self,
            save_audio: bool = True,
            **kwargs: Any,
    ) -> None:
        super().__init__(
            sample_rate=self.SAMPLE_RATE,
            audio_encoding=self.AUDIO_ENCODING,
            **kwargs,
        )

        self._save_audio = save_audio

        from chatterbox.tts_turbo import ChatterboxTurboTTS

        self._model = ChatterboxTurboTTS.from_pretrained(device=self.DEVICE)

    def synthesize(
            self,
            text_stream: Generator[
                str,
                None,
                None,
            ],
    ):
        text = self._read_text_stream(text_stream)

        with CoreTimeMeasure() as core_time_measure:
            with include_measurement(core_time_measure):
                self._timer.maybe_log_time_first_synthesis_request()

                wav = self._model.generate(text).squeeze(0)

                wav = torch.clamp(
                    wav,
                    -1,
                    1,
                ) * INT16_SCALE
                wav = wav.to(torch.int16).numpy()

                self._timer.maybe_log_time_first_audio()

                wav_seconds = len(wav) / self.SAMPLE_RATE
                self._timer.accumulate_audio_seconds(wav_seconds)
                if self._save_audio:
                    self._audio_sink.add(data=wav)

                self._timer.log_time_last_audio()

            self._timer.core_time = core_time_measure.accum_time

    def __str__(self) -> str:
        return f"{self.NAME}"


class KittenSynthesizer(Synthesizer):
    NAME = "Kitten TTS Nano 0.8 INT8"
    SAMPLE_RATE = 24000
    AUDIO_ENCODING = AudioEncodings.INT16
    VOICE_ID = "Bella"
    MODEL_ID = "KittenML/kitten-tts-nano-0.8-int8"
    MAX_SENTENCE_LENGTH = 400

    def __init__(
            self,
            save_audio: bool = True,
            **kwargs: Any,
    ) -> None:
        super().__init__(
            sample_rate=self.SAMPLE_RATE,
            audio_encoding=self.AUDIO_ENCODING,
            **kwargs,
        )

        self._save_audio = save_audio

        from kittentts import KittenTTS

        self._model = KittenTTS(self.MODEL_ID)

    def synthesize(
            self,
            text_stream: Generator[
                str,
                None,
                None,
            ],
    ):
        text = self._read_text_stream(text_stream)

        if len(text) >= self.MAX_SENTENCE_LENGTH:
            self._timer.skip_this_result = True
            print("Text input length reached 400! Kitten-TTS has a limit on sentence length. Skipping this sentence.")
            return

        with CoreTimeMeasure() as core_time_measure:
            with include_measurement(core_time_measure):
                self._timer.maybe_log_time_first_synthesis_request()

                audio = self._model.generate(
                    text,
                    voice=self.VOICE_ID,
                )

                audio = np.clip(
                    audio,
                    -1,
                    1,
                ) * INT16_SCALE
                audio = audio.astype(np.int16)

                self._timer.maybe_log_time_first_audio()

                audio_seconds = len(audio) / self.SAMPLE_RATE
                self._timer.accumulate_audio_seconds(audio_seconds)
                if self._save_audio:
                    self._audio_sink.add(data=audio)

                self._timer.log_time_last_audio()

            self._timer.core_time = core_time_measure.accum_time

    def __str__(self) -> str:
        return f"{self.NAME}"


class PocketSynthesizer(Synthesizer):
    NAME = "Pocket TTS"
    SAMPLE_RATE = 24000
    AUDIO_ENCODING = AudioEncodings.INT16
    SPEAKER_ID = "alba"

    def __init__(
            self,
            save_audio: bool = True,
            **kwargs: Any,
    ) -> None:
        super().__init__(
            sample_rate=self.SAMPLE_RATE,
            audio_encoding=self.AUDIO_ENCODING,
            **kwargs,
        )

        self._save_audio = save_audio

        from pocket_tts import TTSModel

        self._model = TTSModel.load_model()
        self._voice_state = self._model.get_state_for_audio_prompt(self.SPEAKER_ID)

    def synthesize(
            self,
            text_stream: Generator[
                str,
                None,
                None,
            ],
    ):
        text = self._read_text_stream(text_stream)

        with CoreTimeMeasure() as core_time_measure:
            with include_measurement(core_time_measure):
                self._timer.maybe_log_time_first_synthesis_request()

                audio_chunks = self._model.generate_audio_stream(
                    model_state=self._voice_state,
                    text_to_generate=text,
                )

                for chunk in audio_chunks:
                    self._timer.maybe_log_time_first_audio()

                    chunk = torch.clamp(
                        chunk,
                        -1,
                        1,
                    ) * INT16_SCALE
                    chunk = chunk.to(torch.int16).numpy()

                    chunk_seconds = len(chunk) / self.SAMPLE_RATE
                    self._timer.accumulate_audio_seconds(chunk_seconds)
                    if self._save_audio:
                        self._audio_sink.add(data=chunk)

                self._timer.log_time_last_audio()

            self._timer.core_time = core_time_measure.accum_time

    def __str__(self) -> str:
        return f"{self.NAME}"


class NeuTTSNanoSynthesizer(Synthesizer):
    NAME = "NeuTTS Nano"
    SAMPLE_RATE = 24000
    AUDIO_ENCODING = AudioEncodings.INT16
    BACKBONE_ID = "neuphonic/neutts-nano-q4-gguf"
    CODEC_ID = "neuphonic/neucodec-onnx-decoder"

    def __init__(
            self,
            ref_text_path: str,
            ref_codes_path: str,
            save_audio: bool = True,
            **kwargs: Any,
    ) -> None:
        super().__init__(
            sample_rate=self.SAMPLE_RATE,
            audio_encoding=self.AUDIO_ENCODING,
            **kwargs,
        )

        self._save_audio = save_audio

        from neutts import NeuTTS

        self._model = NeuTTS(
            backbone_repo=self.BACKBONE_ID,
            codec_repo=self.CODEC_ID,
        )

        self._ref_text = self._read_if_path(ref_text_path)

        self._ref_codes = torch.load(ref_codes_path)

    @staticmethod
    def _read_if_path(value: str) -> str:
        return open(value, "r", encoding="utf-8").read().strip() if os.path.exists(value) else value

    def synthesize(
            self,
            text_stream: Generator[
                str,
                None,
                None,
            ],
    ):
        input_text = self._read_text_stream(text_stream)

        with CoreTimeMeasure() as core_time_measure:
            with include_measurement(core_time_measure):
                self._timer.maybe_log_time_first_synthesis_request()

                for chunk in self._model.infer_stream(
                        text=input_text,
                        ref_codes=self._ref_codes,
                        ref_text=self._ref_text,
                ):
                    self._timer.maybe_log_time_first_audio()

                    chunk = np.clip(
                        chunk,
                        -1,
                        1,
                    ) * INT16_SCALE
                    chunk = chunk.astype(np.int16)

                    chunk_seconds = len(chunk) / self.SAMPLE_RATE
                    self._timer.accumulate_audio_seconds(chunk_seconds)
                    if self._save_audio:
                        self._audio_sink.add(data=chunk)

                self._timer.log_time_last_audio()

            self._timer.core_time = core_time_measure.accum_time

    def __str__(self) -> str:
        return f"{self.NAME}"


class PiperSynthesizer(Synthesizer):
    NAME = "Piper TTS"
    SAMPLE_RATE = 16000
    AUDIO_ENCODING = AudioEncodings.INT16

    def __init__(
            self,
            model_path: str,
            save_audio: bool = True,
            **kwargs: Any,
    ) -> None:
        super().__init__(
            sample_rate=self.SAMPLE_RATE,
            audio_encoding=self.AUDIO_ENCODING,
            **kwargs,
        )

        self._save_audio = save_audio

        from piper.voice import PiperVoice as Piper

        self._model = Piper.load(model_path)

    def synthesize(
            self,
            text_stream: Generator[
                str,
                None,
                None,
            ],
    ):
        text = self._read_text_stream(text_stream)

        with CoreTimeMeasure() as core_time_measure:
            with include_measurement(core_time_measure):
                self._timer.maybe_log_time_first_synthesis_request()

                chunks = self._model.synthesize(text)

                for chunk in chunks:
                    self._timer.maybe_log_time_first_audio()

                    chunk = np.clip(
                        chunk.audio_float_array,
                        -1,
                        1,
                    ) * INT16_SCALE
                    chunk = chunk.astype(np.int16)

                    chunk_seconds = len(chunk) / self.SAMPLE_RATE
                    self._timer.accumulate_audio_seconds(chunk_seconds)
                    if self._save_audio:
                        self._audio_sink.add(data=chunk)

                self._timer.log_time_last_audio()

            self._timer.core_time = core_time_measure.accum_time

    def __str__(self) -> str:
        return f"{self.NAME}"


class SopranoSynthesizer(Synthesizer):
    NAME = "Soprano TTS"
    SAMPLE_RATE = 32000
    AUDIO_ENCODING = AudioEncodings.INT16
    DEVICE = "cpu"
    BACKEND = "auto"

    def __init__(
            self,
            save_audio: bool = True,
            **kwargs: Any,
    ) -> None:
        super().__init__(
            sample_rate=self.SAMPLE_RATE,
            audio_encoding=self.AUDIO_ENCODING,
            **kwargs,
        )

        self._save_audio = save_audio

        from soprano import SopranoTTS

        self._model = SopranoTTS(
            backend=self.BACKEND,
            device=self.DEVICE,
        )

    def synthesize(
            self,
            text_stream: Generator[
                str,
                None,
                None,
            ],
    ):
        text = self._read_text_stream(text_stream)

        with CoreTimeMeasure() as core_time_measure:
            with include_measurement(core_time_measure):
                self._timer.maybe_log_time_first_synthesis_request()

                stream = self._model.infer_stream(
                    text,
                )

                for chunk in stream:
                    self._timer.maybe_log_time_first_audio()

                    if isinstance(chunk, torch.Tensor):
                        chunk = chunk.detach().cpu()

                    if chunk.dim() == 2 and chunk.shape[0] == 1:
                        chunk = chunk[0]

                    chunk = torch.clamp(
                        chunk,
                        -1,
                        1,
                    ) * INT16_SCALE

                    chunk = chunk.to(torch.int16).numpy()

                    chunk_seconds = len(chunk) / self.SAMPLE_RATE
                    self._timer.accumulate_audio_seconds(chunk_seconds)
                    if self._save_audio:
                        self._audio_sink.add(data=chunk)

                self._timer.log_time_last_audio()

            self._timer.core_time = core_time_measure.accum_time

    def __str__(self) -> str:
        return f"{self.NAME}"


class Supertonic2Synthesizer(Synthesizer):
    NAME = "Supertonic TTS 2"
    SAMPLE_RATE = 44100
    AUDIO_ENCODING = AudioEncodings.INT16
    USE_GPU = False
    LANGUAGE_CODE = "en"
    TOTAL_STEP = 5
    SPEED = 1.05

    def __init__(
            self,
            repo_dir: str,
            onnx_dir: str,
            voice_style_path: str,
            save_audio: bool = True,
            **kwargs: Any,
    ) -> None:
        super().__init__(
            sample_rate=self.SAMPLE_RATE,
            audio_encoding=self.AUDIO_ENCODING,
            **kwargs,
        )

        self._save_audio = save_audio

        import sys
        sys.path.append(repo_dir)
        from py.helper import (
            load_text_to_speech,
            load_voice_style
        )

        self._text_to_speech = load_text_to_speech(
            onnx_dir,
            self.USE_GPU,
        )
        self._style = load_voice_style(
            [voice_style_path],
            verbose=True,
        )

    def synthesize(
            self,
            text_stream: Generator[
                str,
                None,
                None,
            ],
    ):
        text = self._read_text_stream(text_stream)

        with CoreTimeMeasure() as core_time_measure:
            with include_measurement(core_time_measure):
                self._timer.maybe_log_time_first_synthesis_request()

                wav, _ = self._text_to_speech(
                    text,
                    self.LANGUAGE_CODE,
                    self._style,
                    self.TOTAL_STEP,
                    self.SPEED,
                )

                wav = np.clip(
                    wav,
                    -1,
                    1,
                ) * INT16_SCALE
                wav = wav.astype(np.int16)
                wav = np.squeeze(
                    wav,
                    axis=0,
                )

                self._timer.maybe_log_time_first_audio()

                wav_seconds = len(wav) / self.SAMPLE_RATE
                self._timer.accumulate_audio_seconds(wav_seconds)
                if self._save_audio:
                    self._audio_sink.add(data=wav)

                self._timer.log_time_last_audio()

            self._timer.core_time = core_time_measure.accum_time

    def __str__(self) -> str:
        return f"{self.NAME}"


class EspeakNGSynthesizer(Synthesizer):
    NAME = "Espeak NG"
    SAMPLE_RATE = 22050
    AUDIO_ENCODING = AudioEncodings.BYTES
    CHUNK_SIZE_MAX_BYTES = 4096

    def __init__(
            self,
            save_audio: bool = True,
            **kwargs: Any,
    ) -> None:
        super().__init__(
            sample_rate=self.SAMPLE_RATE,
            audio_encoding=self.AUDIO_ENCODING,
            **kwargs,
        )

        self._save_audio = save_audio

    def synthesize(
            self,
            text_stream: Generator[
                str,
                None,
                None,
            ],
    ) -> None:
        text = self._read_text_stream(text_stream)

        with CoreTimeMeasure() as core_time_measure:
            with include_measurement(core_time_measure):
                self._timer.maybe_log_time_first_synthesis_request()

                cmd = [
                    "espeak-ng",
                    "--stdout",
                    text,
                ]

                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0
                )

                while True:
                    chunk = process.stdout.read(self.CHUNK_SIZE_MAX_BYTES)

                    self._timer.maybe_log_time_first_audio()

                    if not chunk:
                        break

                    chunk_seconds = len(chunk) / self.SAMPLE_RATE
                    self._timer.accumulate_audio_seconds(chunk_seconds)
                    if self._save_audio:
                        self._audio_sink.add(data=chunk)

                self._timer.log_time_last_audio()

            self._timer.core_time = core_time_measure.accum_time

    def __str__(self) -> str:
        return f"{self.NAME}"


__all__ = [
    "Synthesizers",
    "Synthesizer",
]
