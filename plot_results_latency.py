import argparse
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (
    FuncFormatter,
    LogLocator,
    MaxNLocator
)

from benchmark import (
    DEFAULT_RESULTS_FOLDER,
    Stats
)
from tts import Synthesizers

Color = Tuple[float, float, float]
DEFAULT_PLOTS_FOLDER = os.path.join(os.path.dirname(__file__), "results/plots")
CAP_VAL = 4000


def rgb_from_hex(x: str) -> Color:
    x = x.strip("# ")
    assert len(x) == 6
    return int(x[:2], 16) / 255, int(x[2:4], 16) / 255, int(x[4:], 16) / 255


BLACK = rgb_from_hex("#000000")
GREY1 = rgb_from_hex("#3F3F3F")
GREY2 = rgb_from_hex("#5F5F5F")
GREY3 = rgb_from_hex("#7F7F7F")
GREY4 = rgb_from_hex("#9F9F9F")
GREY5 = rgb_from_hex("#BFBFBF")
WHITE = rgb_from_hex("#FFFFFF")
BLUE = rgb_from_hex("#377DFF")
COLOR_KOKORO_TTS = rgb_from_hex("#686868")
COLOR_CHATTERBOX_TTS_TURBO = rgb_from_hex("#4F4F4F")
COLOR_KITTEN_TTS = rgb_from_hex("#515151")
COLOR_POCKET_TTS = rgb_from_hex("#808080")
COLOR_NEU_TTS_NANO_Q4_GGUF = rgb_from_hex("#929292")
COLOR_PIPER_TTS = rgb_from_hex("#929292")
COLOR_SOPRANO_TTS = rgb_from_hex("#151515")
COLOR_SUPERTONIC_TTS_2 = rgb_from_hex("#707070")
COLOR_ESPEAK_NG = rgb_from_hex("#363636")

ENGINE_PRINT_NAMES = {
    Synthesizers.AMAZON_POLLY: 'Amazon\nPolly',
    Synthesizers.AZURE_TTS: 'Azure\nTTS',
    Synthesizers.ELEVENLABS: 'ElevenLabs',
    Synthesizers.ELEVENLABS_WEBSOCKET: 'ElevenLabs\nStreaming\nText',
    Synthesizers.OPENAI_TTS: 'OpenAI\nTTS',
    Synthesizers.PICOVOICE_ORCA: 'Picovoice\nOrca',
    Synthesizers.KOKORO_TTS: "Kokoro\nTTS",
    Synthesizers.CHATTERBOX_TTS_TURBO: "Chatterbox\nTTS\nTurbo",
    Synthesizers.KITTEN_TTS: "Kitten\nTTS Nano\n0.8 INT8",
    Synthesizers.POCKET_TTS: "Pocket\nTTS",
    Synthesizers.NEU_TTS_NANO_Q4_GGUF: "Neu TTS\nNano\nQ4 GGUF",
    Synthesizers.PIPER_TTS: "Piper\nTTS",
    Synthesizers.SOPRANO_TTS: "Soprano\nTTS",
    Synthesizers.SUPERTONIC_TTS_2: "Supertonic\nTTS 2",
    Synthesizers.ESPEAK_NG: "ESPEAK\nNG",
}

ENGINE_COLORS = {
    Synthesizers.AMAZON_POLLY: GREY1,
    Synthesizers.AZURE_TTS: GREY2,
    Synthesizers.ELEVENLABS: GREY3,
    Synthesizers.ELEVENLABS_WEBSOCKET: GREY3,
    Synthesizers.OPENAI_TTS: GREY4,
    Synthesizers.PICOVOICE_ORCA: BLUE,
    Synthesizers.KOKORO_TTS: COLOR_KOKORO_TTS,
    Synthesizers.CHATTERBOX_TTS_TURBO: COLOR_CHATTERBOX_TTS_TURBO,
    Synthesizers.KITTEN_TTS: COLOR_KITTEN_TTS,
    Synthesizers.POCKET_TTS: COLOR_POCKET_TTS,
    Synthesizers.NEU_TTS_NANO_Q4_GGUF: COLOR_NEU_TTS_NANO_Q4_GGUF,
    Synthesizers.PIPER_TTS: COLOR_PIPER_TTS,
    Synthesizers.SOPRANO_TTS: COLOR_SOPRANO_TTS,
    Synthesizers.SUPERTONIC_TTS_2: COLOR_SUPERTONIC_TTS_2,
    Synthesizers.ESPEAK_NG: COLOR_ESPEAK_NG,
}


def _plot(
        results_folder: str,
        output_path: str,
        show: bool = False,
        show_error_bars: bool = True,
        only_tts: bool = False,
        no_breakdown: bool = False,
) -> None:
    raw_results = []
    for file in os.listdir(results_folder):
        if file.endswith(".json"):
            json_path = os.path.join(results_folder, file)
            synthesizer, mean, std = Stats.load_results(json_path, scale=1000)
            raw_results.append((synthesizer, mean, std))
    raw_results = [x for x in raw_results if x[0] in ENGINE_PRINT_NAMES.keys()]

    results = []
    for synthesizer in list(ENGINE_COLORS.keys()):
        for raw_result in raw_results:
            if raw_result[0] is synthesizer:
                results.append(raw_result)
                break

    num_results = len(results)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_yscale("linear")

    def round_result(value: float) -> float:
        return round(value, -1)

    def cap_value(x: float) -> Tuple[float, bool]:
        if x > CAP_VAL:
            return CAP_VAL, True
        return x, False

    bottoms = [0.0 for _ in range(num_results)]
    if not only_tts:
        bottoms = []
        rounded_results = []
        colors = []
        for synthesizer, mean, std in results:
            raw_val = round_result(mean.time_to_first_token)
            capped_val, is_capped = cap_value(raw_val)
            rounded_results.append(capped_val)
            colors.append(ENGINE_COLORS[synthesizer])
            bottoms.append(capped_val)
        ax.bar(
            range(num_results),
            rounded_results,
            0.5,
            color=colors,
            label="Time to First Token",
        )
    else:
        bottoms = [0] * num_results

    rounded_results = []
    colors = []
    for i, (synthesizer, mean, std) in enumerate(results):
        rounded_results.append(round_result(mean.first_token_to_speech))
        colors.append(ENGINE_COLORS[synthesizer])
    ax.bar(
        range(num_results),
        rounded_results,
        0.5,
        color=colors,
        bottom=bottoms,
        alpha=0.65 if not only_tts and not no_breakdown else 1.0,
        label="First Token to Speech" if not only_tts else None,
    )

    def scientific_notation(x: float, precision: int = 2) -> str:
        if x == 0:
            return "0"
        coeff, exp = f"{x:.{precision}e}".split("e")
        return f"{float(coeff):.{precision}f}E{int(exp)}"

    def regular_notation(
            x: float,
            precision: int = 1,
            unit: str = "ms",
    ) -> str:
        if x == 0:
            return "0.0"
        mantissa, exp = f"{x:.{precision}e}".split("e")
        exp = int(exp)
        value = float(mantissa) * (10 ** exp)
        decimals = precision - exp

        if decimals > 0:
            result = f"{value:.{decimals}f}"
        else:
            result = str(int(value))

        return result + unit

    has_anomaly = False

    for i, (synthesizer, mean, std) in enumerate(results):
        mean_total_delay_true = mean.voice_assistant_response_time if not only_tts else mean.first_token_to_speech
        std_total_delay = std.voice_assistant_response_time if not only_tts else std.first_token_to_speech

        mean_total_delay_capped, is_capped = cap_value(mean_total_delay_true)

        if is_capped:
            label = f"*{regular_notation(mean_total_delay_true)}"
            has_anomaly = True
        else:
            label = f"{regular_notation(mean_total_delay_true)}"

        string_above_bar = (
            f"{label}" if not show_error_bars
            else f"{label}±{regular_notation(std_total_delay)}"
        )
        ax.text(
            i,
            mean_total_delay_capped + 40,
            string_above_bar,
            ha="center",
            color=BLACK,
            fontsize=8,
        )

    if show_error_bars:
        total_delays = [
            mean.voice_assistant_response_time if not only_tts else mean.first_token_to_speech
            for _, mean, _ in results
        ]
        total_delays_std = [
            std.voice_assistant_response_time if not only_tts else std.first_token_to_speech
            for _, _, std in results
        ]
        plt.errorbar(
            range(num_results),
            total_delays,
            total_delays_std,
            fmt='.',
            color='Black',
            alpha=0.5,
            clip_on=True,
            label="Variability",
        )

    for spine in plt.gca().spines.values():
        if spine.spine_type not in ('bottom', 'left'):
            spine.set_visible(False)

    plt.xticks(
        np.arange(num_results),
        [ENGINE_PRINT_NAMES[x[0]] for x in results],
        fontsize=7,
    )
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda x, _: str(int(x)) + "ms")
    )
    ax.set_ylim(0, CAP_VAL)

    metric = "Voice Assistant Response Time" if not only_tts else "First Token to Speech"
    plt.suptitle(f"{metric}", fontsize=20, x=0.51, y=0.96)

    if has_anomaly:
        plt.figtext(
            0.99, 0.91,
            "* Values are capped at 4000 ms for readability.\n  The actual values are shown above the bars.",
            ha="right",
            fontsize=9,
            color=BLACK,
        )

    if (not only_tts or show_error_bars) and not no_breakdown:
        ax.legend(loc="upper left", fontsize=12, framealpha=0)

    if show_error_bars:
        output_path = output_path.replace(".png", "_error_bars.png")
    if no_breakdown:
        output_path = output_path.replace(".png", "_no_breakdown.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved plot to `{output_path}`")

    if show:
        plt.show()
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-folder",
        default=DEFAULT_RESULTS_FOLDER,
        help="Path to results folder")
    parser.add_argument("--show-errors", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-breakdown", action="store_true")
    args = parser.parse_args()

    _plot(
        results_folder=args.results_folder,
        output_path=os.path.join(DEFAULT_PLOTS_FOLDER, "voice_assistant_response_time.png"),
        show=args.show,
        show_error_bars=args.show_errors,
        only_tts=False,
        no_breakdown=args.no_breakdown,
    )
    _plot(
        results_folder=args.results_folder,
        output_path=os.path.join(DEFAULT_PLOTS_FOLDER, "first_token_to_speech.png"),
        show=args.show,
        show_error_bars=args.show_errors,
        only_tts=True,
        no_breakdown=args.no_breakdown,
    )


if __name__ == "__main__":
    main()
