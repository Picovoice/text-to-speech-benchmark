from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional
import psutil
import threading
import time


@dataclass
class Timer:
    _time_llm_request: Optional[float] = None
    _time_first_llm_token: Optional[float] = None
    _time_last_llm_token: Optional[float] = None
    _time_first_synthesis_request: Optional[float] = None
    _time_first_audio: Optional[float] = None
    _time_last_audio: Optional[float] = None
    _core_time: float = 0.0
    _accumulated_audio_seconds: float = 0.0
    _num_tokens: int = 0
    _skip_this_result: bool = False

    @staticmethod
    def _get_time() -> float:
        return time.perf_counter()

    def log_time_llm_request(self) -> None:
        self._time_llm_request = self._get_time()

    def maybe_log_time_first_llm_token(self) -> None:
        if self._time_first_llm_token is None:
            self._time_first_llm_token = self._get_time()

    def maybe_log_time_first_synthesis_request(self) -> None:
        if self._time_first_synthesis_request is None:
            self._time_first_synthesis_request = self._get_time()

    def maybe_set_time_first_synthesis_request(self, seconds: float) -> None:
        if self._time_first_synthesis_request is None:
            self._time_first_synthesis_request = seconds

    def log_time_first_synthesis_request(self) -> None:
        self._time_first_synthesis_request = self._get_time()

    def log_time_last_llm_token(self) -> None:
        self._time_last_llm_token = self._get_time()

    def maybe_log_time_first_audio(self) -> None:
        if self._time_first_audio is None:
            self._time_first_audio = self._get_time()

    def log_time_last_audio(self) -> None:
        self._time_last_audio = self._get_time()

    def increment_num_tokens(self) -> None:
        self._num_tokens += 1

    @property
    def num_tokens(self) -> int:
        return self._num_tokens

    def first_token_to_speech(self) -> Optional[float]:
        if self._time_first_audio is not None and self._time_first_llm_token is not None:
            return self._time_first_audio - self._time_first_llm_token
        else:
            return None

    def time_to_first_token(self) -> Optional[float]:
        if self._time_first_llm_token is not None and self._time_llm_request is not None:
            return self._time_first_llm_token - self._time_llm_request
        else:
            return None

    def tts_process_seconds(self) -> Optional[float]:
        if self._time_first_audio is not None and self._time_first_synthesis_request is not None:
            return self._time_first_audio - self._time_first_synthesis_request
        else:
            return None

    def llm_text_generation_seconds(self) -> Optional[float]:
        if self._time_last_llm_token is not None and self._time_first_llm_token is not None:
            return self._time_last_llm_token - self._time_first_llm_token
        else:
            return None

    def voice_assistant_response_time(self) -> Optional[float]:
        ftts = self.first_token_to_speech()
        ttft = self.time_to_first_token()
        if ftts is not None and ttft is not None:
            return ftts + ttft
        else:
            return None

    def num_tokens_per_second(self) -> Optional[float]:
        if self._time_last_llm_token is not None and self._time_first_llm_token is not None:
            return self._num_tokens / (self._time_last_llm_token - self._time_first_llm_token)
        else:
            return None

    def wait_for_first_audio(self) -> None:
        while self._time_first_audio is None and not self._skip_this_result:
            time.sleep(0.01)

    def wait_for_last_audio(self) -> None:
        while self._time_last_audio is None:
            time.sleep(0.01)

    def reset(self) -> None:
        self._time_llm_request = None
        self._time_first_llm_token = None
        self._time_last_llm_token = None
        self._time_first_synthesis_request = None
        self._time_first_audio = None
        self._time_last_audio = None

        self._core_time = 0.0
        self._accumulated_audio_seconds = 0.0

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

    @property
    def accumulated_audio_seconds(self) -> float:
        return self._accumulated_audio_seconds

    def accumulate_audio_seconds(
            self,
            audio_seconds: float,
    ) -> None:
        self._accumulated_audio_seconds += audio_seconds

    @property
    def core_time(self) -> float:
        return self._core_time

    @core_time.setter
    def core_time(
            self,
            val: float,
    ):
        self._core_time = val


class CoreTimeMeasure:
    def __init__(self):
        self._proc_main = psutil.Process()
        self._paused = True
        self._pause_start = 0
        self._pause_end = 0
        self._accum_time = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return

    @property
    def accum_time(self) -> float:
        return self._accum_time

    def pause(self):
        assert not self._paused
        self._pause_end = self._core_time_tree()
        self._paused = True
        self._accum_time += self._pause_end - self._pause_start

    def resume(self):
        assert self._paused
        self._pause_start = self._core_time_tree()
        self._paused = False

    def _core_time_tree(self):
        total = 0.0
        proc_list = [self._proc_main] + self._proc_main.children(recursive=True)
        for p in proc_list:
            try:
                t = p.cpu_times()
                total += t.user + t.system
            except psutil.NoSuchProcess:
                pass
        return total


@contextmanager
def include_measurement(*measurements):
    for m in measurements:
        m.resume()
    try:
        yield
    finally:
        for m in measurements:
            m.pause()


def _memory_tree(proc_main):
    total = 0
    proc_list = [proc_main] + proc_main.children(recursive=True)
    for p in proc_list:
        try:
            total += p.memory_full_info().pss
        except psutil.NoSuchProcess:
            pass
    return total


@contextmanager
def measure_peak_memory(
        interval=0.05,
):
    proc_main = psutil.Process()
    peak_mem = 0
    result = {
        "peak_mem": peak_mem,
    }
    stop_event = threading.Event()

    initial_mem = _memory_tree(proc_main)

    def _measure():
        nonlocal peak_mem
        while not stop_event.is_set():
            mem = _memory_tree(proc_main)
            peak_mem = max(peak_mem, mem)
            stop_event.wait(interval)

    t = threading.Thread(target=_measure)
    t.start()

    try:
        yield result
    finally:
        stop_event.set()
        t.join()

    result["peak_mem"] = (peak_mem - initial_mem) / (1024**2)


__all__ = [
    "Timer",
    "measure_peak_memory",
    "CoreTimeMeasure",
    "include_measurement",
]
