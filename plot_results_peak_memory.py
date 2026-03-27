import argparse
import json
import os
from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.ticker import FuncFormatter

from benchmark import DEFAULT_RESULTS_FOLDER
from tts import Synthesizers

Color = Tuple[float, float, float]
DEFAULT_PLOTS_FOLDER = os.path.join(os.path.dirname(__file__), "results/plots")


def rgb_from_hex(x: str) -> Color:
    x = x.strip("# ")
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
MODELS_STRING = [k.value for k in ENGINE_PRINT_NAMES.keys()]


def _plot(
        results_folder: str,
        output_path: str,
) -> None:
    mean_mem = []
    for model in MODELS_STRING:
        model_values = []
        for file in os.listdir(results_folder):
            if not file.endswith(".json") or not file.startswith(f"results_tts_{model}"):
                continue
            filepath = os.path.join(results_folder, file)
            with open(filepath) as f:
                data = json.load(f)
            if "peak_memory_dict" not in data:
                continue
            peak_dict = data["peak_memory_dict"]
            values = [v for k, v in peak_dict.items()]
            model_values.extend(values)
        avg_memory = np.mean(model_values) if model_values else 0
        mean_mem.append(avg_memory)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        range(len(MODELS)),
        mean_mem,
        width=0.5,
        color=[ENGINE_COLORS[m] for m in MODELS],
        alpha=1.0
    )

    def scientific_notation(x: float, precision: int = 2) -> str:
        coeff, exp = f"{x:.{precision}e}".split("e")
        if int(exp) == 0:
            return f"{float(coeff):.{precision}f}"
        return f"{float(coeff):.{precision}f}E{int(exp)}"

    for i, val in enumerate(mean_mem):
        ax.text(i, val * 1.05, scientific_notation(float(val)), ha="center", fontsize=10, color=BLACK)

    def log_e_formatter(x, pos):
        if x == 0:
            return "0"
        exp = int(np.floor(np.log10(x)))
        return f"1E{exp}"

    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([ENGINE_PRINT_NAMES[m] for m in MODELS], fontsize=10)
    ax.set_yscale("log", base=10)
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=[1]))
    ax.yaxis.set_major_formatter(FuncFormatter(log_e_formatter))
    ax.set_ylabel("Peak Memory (MB)", fontsize=14)

    for spine in ax.spines.values():
        if spine.spine_type not in ["bottom", "left"]:
            spine.set_visible(False)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved bar plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-folder",
        default=DEFAULT_RESULTS_FOLDER,
        help="Path to results folder",
    )
    args = parser.parse_args()

    output_path = os.path.join(
        DEFAULT_PLOTS_FOLDER,
        "peak_memory.png",
    )
    _plot(
        results_folder=args.results_folder,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
