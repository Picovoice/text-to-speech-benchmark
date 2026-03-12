import time
from dataclasses import dataclass


@dataclass
class Timer:
    _time_llm_request: float = -1.0
    _time_first_llm_token: float = -1.0
    _time_last_llm_token: float = -1.0
    _time_first_synthesis_request: float = -1.0
    _time_first_audio: float = -1.0
    _time_last_audio: float = -1.0

    _num_tokens: int = 0
    _skip_this_result: bool = False

    @staticmethod
    def _get_time() -> float:
        return time.perf_counter()

    def log_time_llm_request(self) -> None:
        self._time_llm_request = self._get_time()

    def maybe_log_time_first_llm_token(self) -> None:
        if self._time_first_llm_token == -1.0:
            self._time_first_llm_token = self._get_time()

    def maybe_log_time_first_synthesis_request(self) -> None:
        if self._time_first_synthesis_request == -1.0:
            self._time_first_synthesis_request = self._get_time()

    def maybe_set_time_first_synthesis_request(self, seconds: float) -> None:
        if self._time_first_synthesis_request == -1.0:
            self._time_first_synthesis_request = seconds

    def log_time_first_synthesis_request(self) -> None:
        self._time_first_synthesis_request = self._get_time()

    def log_time_last_llm_token(self) -> None:
        self._time_last_llm_token = self._get_time()

    def maybe_log_time_first_audio(self) -> None:
        if self._time_first_audio == -1.0:
            self._time_first_audio = self._get_time()

    def log_time_last_audio(self) -> None:
        self._time_last_audio = self._get_time()

    def increment_num_tokens(self) -> None:
        self._num_tokens += 1

    def first_token_to_speech(self) -> float:
        return self._time_first_audio - self._time_first_llm_token

    def time_to_first_token(self) -> float:
        return self._time_first_llm_token - self._time_llm_request

    def tts_process_seconds(self) -> float:
        return self._time_first_audio - self._time_first_synthesis_request

    def llm_text_generation_seconds(self) -> float:
        return self._time_last_llm_token - self._time_first_llm_token

    def voice_assistant_response_time(self) -> float:
        return self.first_token_to_speech() + self.time_to_first_token()

    def num_tokens_per_second(self) -> float:
        return self._num_tokens / (self._time_last_llm_token - self._time_first_llm_token)

    def wait_for_first_audio(self) -> None:
        while self._time_first_audio == -1.0 and not self._skip_this_result:
            time.sleep(0.01)

    def wait_for_last_audio(self) -> None:
        while self._time_last_audio == -1.0:
            time.sleep(0.01)

    def reset(self) -> None:
        self._time_llm_request = -1.0
        self._time_first_llm_token = -1.0
        self._time_last_llm_token = -1.0
        self._time_first_synthesis_request = -1.0
        self._time_first_audio = -1.0
        self._time_last_audio = -1.0

        self._num_tokens = 0
        self._skip_this_result = False

    @property
    def skip_this_result(self) -> bool:
        return self._skip_this_result

    @skip_this_result.setter
    def skip_this_result(
            self,
            val: bool,
    ):
        self._skip_this_result = val


__all__ = [
    "Timer",
]
