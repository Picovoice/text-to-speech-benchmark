import os
import json
import matplotlib.pyplot as plt

# Directory containing json files
DATA_DIR = "/home/pear/work/github/tts-latency-benchmark/results/data/"

# Directory to save the figure
OUTPUT_DIR = "/home/pear/work/github/tts-latency-benchmark/results/plots/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "peak_memory_vs_input_length.png")

# List of model names
models = [
    "picovoice_orca",
    "kokoro_tts",
    "espeak_ng",
    "kitten_tts",
    "pocket_tts",
    "soprano_tts",
    "chatterbox_tts_turbo",
    "supertonic_tts_2",
    "neu_tts_nano_q4_gguf",
    "piper_tts",
]

plt.figure()

for file in os.listdir(DATA_DIR):
    if not file.endswith(".json"):
        continue

    filepath = os.path.join(DATA_DIR, file)

    for model in models:
        if file.startswith("results_tts_" + model):
            with open(filepath, "r") as f:
                data = json.load(f)

            if "peak_memory_dict" in data:
                peak_dict = data["peak_memory_dict"]

                # Sort input lengths
                x = sorted(int(k) for k in peak_dict.keys())
                y = [peak_dict[str(i)] for i in x]

                plt.plot(x, y, marker="o", label=model)

plt.xlabel("Input Length (number of input characters)")
plt.ylabel("Peak Memory (MB)")
plt.title("Peak Memory vs Input Length")

plt.yscale("log", base=10)

plt.legend()
plt.grid(True)

# Save to PNG
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")

print(f"Plot saved to {OUTPUT_PATH}")
