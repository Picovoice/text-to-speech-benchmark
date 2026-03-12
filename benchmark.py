import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Sequence,
    Tuple,
)

import matplotlib.pyplot as plt
from tqdm import tqdm

from data import TextDatasets, TextDataset
from llm import LLMs, LLM
from tts import (
    Synthesizers,
    Synthesizer,
    Timer,
)

DEFAULT_RESULTS_FOLDER = os.path.join(os.path.dirname(__file__), "results", "data")
DEFAULT_DATASET = TextDatasets.TASKMASTER2


@dataclass
class TimingResult:
    voice_assistant_response_time: float
    time_to_first_token: float
    first_token_to_speech: float
    tts_process_seconds: float
    num_words: int
    num_tokens_per_second: float

    @staticmethod
    def _compute_statistics(results: Sequence['TimingResult'], fn: Callable) -> 'TimingResult':
        return TimingResult(
            voice_assistant_response_time=fn([r.voice_assistant_response_time for r in results]),
            time_to_first_token=fn([r.time_to_first_token for r in results]),
            first_token_to_speech=fn([r.first_token_to_speech for r in results]),
            tts_process_seconds=fn([r.tts_process_seconds for r in results]),
            num_words=int(fn([r.num_words for r in results])),
            num_tokens_per_second=fn([r.num_tokens_per_second for r in results]),
        )

    @classmethod
    def mean_from_results(cls, results: Sequence['TimingResult']) -> 'TimingResult':
        if len(results) == 0:
            print("WARNING: Cannot compute mean of empty list")
            return TimingResult(0, 0, 0, 0, 0, 0)

        def _mean(values: Sequence[Any]) -> float:
            return sum(values) / len(values)

        return cls._compute_statistics(results, _mean)

    @classmethod
    def std_from_results(cls, results: Sequence['TimingResult']) -> 'TimingResult':
        if len(results) == 0:
            print("WARNING: Cannot compute standard deviation of empty list")
            return TimingResult(0, 0, 0, 0, 0, 0)

        def _std(values):
            mean = sum(values) / len(values)
            return (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5

        return cls._compute_statistics(results, _std)


class Stats:
    MAX_LLM_DELAY_SECONDS = 0.6

    def __init__(self, tts: Synthesizers, results_folder: Optional[str] = None) -> None:
        self._results = []

        self._tts_type_string = tts.value

        self._output_folder = os.path.join(results_folder or DEFAULT_RESULTS_FOLDER)
        os.makedirs(self._output_folder, exist_ok=True)

    def accumulate(self, timing_result: TimingResult) -> None:
        self._results.append(timing_result)

    def _filter_outliers(self, results: Sequence[TimingResult]) -> Sequence[TimingResult]:
        filtered_results = []
        for result in results:
            if result.time_to_first_token > self.MAX_LLM_DELAY_SECONDS:
                continue

            filtered_results.append(result)

        return filtered_results

    def save_results(self) -> None:
        results = self._filter_outliers(self._results)

        num_sentences = len(results)

        mean = TimingResult.mean_from_results(results)
        std = TimingResult.std_from_results(results)

        print("Summary statistics:")
        print(f"Total number of sentences: {num_sentences}")
        print(
            "Voice Assistant Response Time: "
            f"{mean.voice_assistant_response_time:.2f} +- {std.first_token_to_speech:.2f} s")
        print(f"Time to First Token: {mean.time_to_first_token:.2f} +- {std.time_to_first_token:.2f} s")
        print(f"First Token to Speech: {mean.first_token_to_speech:.2f} +- {std.first_token_to_speech:.2f} s")
        print(f"TTS processing time: {mean.tts_process_seconds:.2f} +- {std.tts_process_seconds:.2f} s")
        print(f"Mean number of words per sentence: {mean.num_words:.1f} +- {std.num_words:.1f}")
        print(f"Mean tokens per second: {mean.num_tokens_per_second:.2f} +- {std.num_tokens_per_second:.2f}")

        fig, axs = plt.subplots(3, 2, figsize=(14, 8))
        axs[0, 0].hist([r.voice_assistant_response_time for r in self._results], bins=10)
        axs[0, 0].set_title('voice_assistant_response_time')
        axs[0, 1].hist([r.time_to_first_token for r in self._results], bins=10)
        axs[0, 1].set_title('time_to_first_token')
        axs[0, 1].axvline(x=self.MAX_LLM_DELAY_SECONDS, color='r', linestyle='--')
        axs[1, 0].hist([r.first_token_to_speech for r in self._results], bins=10)
        axs[1, 0].set_title('first_token_to_speech')
        axs[1, 1].hist([r.num_words for r in self._results], bins=10)
        axs[1, 1].set_title('num_words')
        axs[2, 0].hist([r.num_tokens_per_second for r in self._results], bins=10)
        axs[2, 0].set_title('num_tokens_per_second')
        axs[2, 1].hist([r.tts_process_seconds for r in self._results], bins=10)
        axs[2, 1].set_title('tts_process_seconds')

        output_path = os.path.join(self._output_folder, f"hists_tts_{self._tts_type_string}.png")
        plt.savefig(output_path)
        plt.close()

        results_json_path = os.path.join(self._output_folder, f"results_tts_{self._tts_type_string}.json")
        results_dict = {
            "total_sentences": num_sentences,
            "mean_voice_assistant_response_time": mean.voice_assistant_response_time,
            "mean_time_to_first_token": mean.time_to_first_token,
            "mean_first_token_to_speech": mean.first_token_to_speech,
            "mean_tts_process_seconds": mean.tts_process_seconds,
            "mean_num_words": mean.num_words,
            "mean_num_tokens_per_second": mean.num_tokens_per_second,
            "std_voice_assistant_response_time": std.voice_assistant_response_time,
            "std_time_to_first_token": std.time_to_first_token,
            "std_first_token_to_speech": std.first_token_to_speech,
            "std_tts_process_seconds": std.tts_process_seconds,
            "std_num_words": std.num_words,
            "std_num_tokens_per_second": std.num_tokens_per_second,
        }
        with open(results_json_path, "w") as f:
            json.dump(results_dict, f, indent=4)

        print("Results saved to:", self._output_folder)

    @staticmethod
    def load_results(json_path: str, scale: float = 1.0) -> Tuple[Synthesizers, TimingResult, TimingResult]:
        tts_type_string = None
        for synthesizer in Synthesizers:
            if synthesizer.value in json_path:
                tts_type_string = synthesizer.value
        if tts_type_string is None:
            raise ValueError(f"Could not determine TTS type from path: `{json_path}`")

        with open(json_path, "r") as f:
            results_dict = json.load(f)

        mean = TimingResult(
            voice_assistant_response_time=results_dict["mean_voice_assistant_response_time"] * scale,
            time_to_first_token=results_dict["mean_time_to_first_token"] * scale,
            first_token_to_speech=results_dict["mean_first_token_to_speech"] * scale,
            tts_process_seconds=results_dict["mean_tts_process_seconds"] * scale,
            num_words=results_dict["mean_num_words"] * scale,
            num_tokens_per_second=results_dict["mean_num_tokens_per_second"] * scale)
        std = TimingResult(
            voice_assistant_response_time=results_dict["std_voice_assistant_response_time"] * scale,
            time_to_first_token=results_dict["std_time_to_first_token"] * scale,
            first_token_to_speech=results_dict["std_first_token_to_speech"] * scale,
            tts_process_seconds=results_dict["std_tts_process_seconds"] * scale,
            num_words=results_dict["std_num_words"] * scale,
            num_tokens_per_second=results_dict["std_num_tokens_per_second"] * scale)

        return Synthesizers(tts_type_string), mean, std


def get_default_llm_type(tts_type: Synthesizers) -> LLMs:
    return LLMs.PICOLLM


def get_llm_init_kwargs(args: argparse.Namespace) -> Dict[str, str]:
    kwargs = dict()
    llm_type = get_default_llm_type(Synthesizers(args.synthesizer))

    if llm_type is LLMs.PICOLLM:
        kwargs["access_key"] = args.picovoice_access_key
        kwargs["model_path"] = args.picollm_model_path
        kwargs["device"] = args.picollm_device

    return kwargs


def get_synthesizer_init_kwargs(args: argparse.Namespace) -> Dict[str, str]:
    kwargs = dict()
    synthesizer_type = Synthesizers(args.synthesizer)

    if synthesizer_type is Synthesizers.PICOVOICE_ORCA:
        if args.picovoice_access_key is None:
            raise ValueError(
                "Picovoice access key is required when using Picovoice TTS. Specify with `--picovoice-access-key`.")
        kwargs["access_key"] = args.picovoice_access_key
        kwargs["model_path"] = args.orca_model_path
        kwargs["device"] = args.orca_device
        kwargs["library_path"] = args.orca_library_path

    elif synthesizer_type is Synthesizers.AZURE_TTS:
        if args.azure_speech_key is None or args.azure_speech_region is None:
            raise ValueError(
                "Azure speech key and region are required when using Azure TTS. "
                "Specify with `--azure-speech-key` and `--azure-speech-region`.")
        kwargs['speech_key'] = args.azure_speech_key
        kwargs['speech_region'] = args.azure_speech_region

    elif synthesizer_type is Synthesizers.AMAZON_POLLY:
        if args.aws_profile_name is None:
            raise ValueError(
                "AWS profile name is required when using AWS Polly. Specify with `--aws-profile-name`.")
        kwargs["aws_profile_name"] = args.aws_profile_name

    elif synthesizer_type is Synthesizers.ELEVENLABS or synthesizer_type is Synthesizers.ELEVENLABS_WEBSOCKET:
        if args.elevenlabs_api_key is None:
            raise ValueError(
                "Elevenlabs API key is required when using Elevenlabs TTS. Specify with `--elevenlabs-api-key`.")
        kwargs["api_key"] = args.elevenlabs_api_key

    elif synthesizer_type is Synthesizers.OPENAI_TTS:
        if args.openai_api_key is None:
            raise ValueError(
                f"An OpenAI access key is required when using OpenAI models. Specify with `--openai-api-key`.")
        kwargs["api_key"] = args.openai_api_key

    return kwargs


async def _run_benchmark_iteration(
        llm: LLM,
        synthesizer: Synthesizer,
        sentence: str,
        timer: Timer,
        stats: Stats,
        results_folder: str,
        verbose: bool,
        counter: int) -> None:
    timer.reset()

    timer.log_time_llm_request()

    if synthesizer.is_async:
        await synthesizer.synthesize_async(text_stream=llm.query_async(sentence))
    else:
        synthesizer.synthesize(text_stream=llm.query(sentence))

    timer.wait_for_first_audio()

    if not timer.skip_this_result:
        timing_result = TimingResult(
            voice_assistant_response_time=timer.voice_assistant_response_time(),
            time_to_first_token=timer.time_to_first_token(),
            first_token_to_speech=timer.first_token_to_speech(),
            tts_process_seconds=timer.tts_process_seconds(),
            num_words=len(llm.last_response.split()),
            num_tokens_per_second=timer.num_tokens_per_second())
        stats.accumulate(timing_result=timing_result)

        if verbose:
            print(f"Question: {sentence}")
            print(f"LLM response: {llm.last_response}")
            print(f"Voice Assistant Response Time: {timing_result.voice_assistant_response_time:.2f} s")
            print(f"Time to First Token: {timing_result.time_to_first_token:.2f} s")
            print(f"First Token to Speech: {timing_result.first_token_to_speech:.2f} s")
            timer.wait_for_last_audio()
            audio_path = os.path.join(results_folder, f"audio_{counter}.wav")
            synthesizer.save_and_reset_last_audio(audio_path)
            print(f"Saved audio to `{audio_path}`")
            print()
    else:
        if verbose:
            print("Error occurred for this sentence. Skip the result.")



async def main(args: argparse.Namespace) -> None:
    num_interactions = args.num_interactions
    results_folder = args.results_folder
    verbose = args.verbose
    tts_type = Synthesizers(args.synthesizer)
    llm_type = get_default_llm_type(tts_type)

    dataset = TextDataset.create(DEFAULT_DATASET)

    timer = Timer()

    synthesizer_init_kwargs = get_synthesizer_init_kwargs(args)
    synthesizer = Synthesizer.create(
        Synthesizers(args.synthesizer),
        timer=timer,
        **synthesizer_init_kwargs)

    llm_init_kwargs = get_llm_init_kwargs(args)
    llm = LLM.create(llm_type, **llm_init_kwargs)

    benchmark_sentences = dataset.get_random_sentences(num=num_interactions)

    stats = Stats(tts=tts_type, results_folder=results_folder)

    counter = 0
    print("Running benchmark ...")
    for sentence in tqdm(benchmark_sentences):
        await _run_benchmark_iteration(
            llm=llm,
            synthesizer=synthesizer,
            sentence=sentence,
            timer=timer,
            stats=stats,
            results_folder=results_folder,
            verbose=verbose,
            counter=counter)
        counter += 1

    stats.save_results()

    synthesizer.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--engine",
        dest="synthesizer",
        default=Synthesizers.PICOVOICE_ORCA.value,
        choices=[s.value for s in Synthesizers],
        help="Choose voice synthesizer to use")
    parser.add_argument(
        "--picovoice-access-key",
        required=True,
        help="AccessKey obtained from Picovoice Console")
    parser.add_argument(
        "--picollm-model-path",
        required=True,
        help="PicoLLM model obtained from Picovoice Console")
    parser.add_argument(
        "--picollm-device",
        default=None,
        help="Device for picollm to use")
    parser.add_argument(
        "--orca-model-path",
        default=None,
        help="Path to the model parameters file")
    parser.add_argument(
        "--orca-library-path",
        default=None,
        help="Path to Orca's dynamic library")
    parser.add_argument(
        "--orca-device",
        default=None,
        help="Device for orca to use")

    parser.add_argument(
        "--openai-api-key",
        default=None,
        help="Open AI API key. Needed when using openai models")

    parser.add_argument(
        "--aws-profile-name",
        default=None,
        help="AWS profile name to use for AWS Polly")

    parser.add_argument(
        "--azure-speech-key",
        default=None,
        help="Azure access token")
    parser.add_argument(
        "--azure-speech-region",
        default=None,
        help="Azure speech location")

    parser.add_argument(
        "--elevenlabs-api-key",
        default=None,
        help="Elevenlabs API key")

    parser.add_argument(
        "--num-interactions",
        type=int,
        default=200,
        help="Number of interactions to benchmark")
    parser.add_argument(
        "--results-folder",
        default=DEFAULT_RESULTS_FOLDER,
        help="Folder to save results")

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output")

    asyncio.run(main(parser.parse_args()))
