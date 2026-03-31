import argparse
import os
from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.ticker import FuncFormatter

from benchmark import (
    DEFAULT_RESULTS_FOLDER,
    Stats
)
from tts import Synthesizers

Color = Tuple[float, float, float]
DEFAULT_PLOTS_FOLDER = os.path.join(os.path.dirname(__file__), "results/plots")


def rgb_from_hex(x: str) -> Color:
    x = x.strip("# ")
    assert len(x) == 6
    return int(x[:2], 16) / 255, int(x[2:4], 16) / 255, int(x[4:], 16) / 255


BLACK = rgb_from_hex("#000000")
BLUE = rgb_from_hex("#377DFF")
COLOR_KOKORO_TTS = rgb_from_hex("#686868")
COLOR_CHATTERBOX_TTS_TURBO = rgb_from_hex("#4F4F4F")
COLOR_KITTEN_TTS = rgb_from_hex("#515151")
COLOR_POCKET_TTS = rgb_from_hex("#808080")
COLOR_NEU_TTS_NANO_Q4_GGUF = rgb_from_hex("#929292")
COLOR_PIPER_TTS = rgb_from_hex("#929292")
COLOR_SOPRANO_TTS = rgb_from_hex("#151515")
COLOR_SUPERTONIC_TTS_2 = rgb_from_hex("#707070")

ENGINE_PRINT_NAMES = {
    Synthesizers.PICOVOICE_ORCA: 'Picovoice\nOrca',
    Synthesizers.KOKORO_TTS: "Kokoro\nTTS",
    Synthesizers.CHATTERBOX_TTS_TURBO: "Chatterbox\nTTS\nTurbo",
    Synthesizers.KITTEN_TTS: "Kitten\nTTS Nano\n0.8 INT8",
    Synthesizers.POCKET_TTS: "Pocket\nTTS",
    Synthesizers.NEU_TTS_NANO_Q4_GGUF: "Neu TTS\nNano\nQ4 GGUF",
    Synthesizers.PIPER_TTS: "Piper\nTTS",
    Synthesizers.SOPRANO_TTS: "Soprano\nTTS",
    Synthesizers.SUPERTONIC_TTS_2: "Supertonic\nTTS 2",
}

ENGINE_COLORS = {
    Synthesizers.PICOVOICE_ORCA: BLUE,
    Synthesizers.KOKORO_TTS: COLOR_KOKORO_TTS,
    Synthesizers.CHATTERBOX_TTS_TURBO: COLOR_CHATTERBOX_TTS_TURBO,
    Synthesizers.KITTEN_TTS: COLOR_KITTEN_TTS,
    Synthesizers.POCKET_TTS: COLOR_POCKET_TTS,
    Synthesizers.NEU_TTS_NANO_Q4_GGUF: COLOR_NEU_TTS_NANO_Q4_GGUF,
    Synthesizers.PIPER_TTS: COLOR_PIPER_TTS,
    Synthesizers.SOPRANO_TTS: COLOR_SOPRANO_TTS,
    Synthesizers.SUPERTONIC_TTS_2: COLOR_SUPERTONIC_TTS_2,
}


MODELS = list(ENGINE_PRINT_NAMES.keys())


def _plot(
        results_folder: str,
        output_path: str,
) -> None:
    raw_results = []
    for file in os.listdir(results_folder):
        for synthesizer_type in MODELS:
            synthesizer_name = synthesizer_type.value
            if synthesizer_name in file and file.endswith(".json"):
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

    max_delay = 0.0
    for synthesizer, mean, std in results:
        print(
            f"TTS: {synthesizer.value}")
        print(f"Core hour ratio: {mean.core_hour_ratio:.0f} +- {std.core_hour_ratio:.0f}")
        max_delay = max(max_delay, mean.core_hour_ratio)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_yscale("log", base=10)

    bottoms = [0 for _ in range(num_results)]

    rounded_results = []
    colors = []
    for i, (synthesizer, mean, std) in enumerate(results):
        rounded_results.append(mean.core_hour_ratio)
        colors.append(ENGINE_COLORS[synthesizer])
    ax.bar(
        range(num_results),
        rounded_results,
        0.5,
        color=colors,
        bottom=bottoms,
        alpha=1.0,
        label=None,
    )

    def regular_notation(
            x: float,
            precision: int = 1,
            unit: str = "×",
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

    total_delays = []
    total_delays_std = []
    for i, (synthesizer, mean, std) in enumerate(results):
        mean_total_delay = mean.core_hour_ratio
        std_total_delay = std.core_hour_ratio
        rounded_result = mean_total_delay
        total_delays.append(rounded_result)
        std_total_delay = std.core_hour_ratio
        total_delays_std.append(std_total_delay)
        ax.text(
            i,
            rounded_result * 1.05,
            regular_notation(rounded_result),
            ha="center",
            color=BLACK,
            fontsize=10,
        )

    for spine in plt.gca().spines.values():
        if spine.spine_type != 'bottom' and spine.spine_type != 'left':
            spine.set_visible(False)

    y_max = max_delay + (max_delay / 6)
    plt.ylim(0, y_max)
    plt.xticks(
        np.arange(
            0,
            len(rounded_results),
        ),
        [ENGINE_PRINT_NAMES[x[0]] for x in results],
        fontsize=10,
    )
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10))

    def y_axis_formatter(x, pos):
        return f"{x:.0f}×"

    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    metric = "Core Hour Ratio"
    plt.suptitle(f"{metric}", fontsize=20, x=0.51, y=0.96)
    ax.set_title("(log₁₀ scale)", fontsize=12, pad=6)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved plot to `{output_path}`")

    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-folder",
        default=DEFAULT_RESULTS_FOLDER,
        help="Path to results folder")
    args = parser.parse_args()

    output_path = os.path.join(
        DEFAULT_PLOTS_FOLDER,
        "core_hour_ratio.png",
    )
    _plot(
        results_folder=args.results_folder,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
