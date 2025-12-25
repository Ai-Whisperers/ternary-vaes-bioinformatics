import json
import os
import subprocess
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
VSCODE_DIR = PROJECT_ROOT / ".vscode"
SETTINGS_FILE = VSCODE_DIR / "settings.json"


def run_command(command, description):
    print(f"\nrunning {description}...")
    try:
        subprocess.run(command, check=True, shell=True, cwd=PROJECT_ROOT)
        print(f"✅ {description} complete.")
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed or found issues.")


def run_formatter():
    run_command("black .", "Black (Formatter)")
    run_command("isort .", "isort (Import Sorter)")


def run_linter():
    # standard pylint run
    run_command("pylint src scripts tests --rcfile=.pylintrc", "Pylint")


def add_words_to_dictionary(words):
    if not SETTINGS_FILE.exists():
        print(f"Creating {SETTINGS_FILE}")
        VSCODE_DIR.mkdir(exist_ok=True)
        settings = {"cSpell.words": []}
    else:
        try:
            with open(SETTINGS_FILE, "r") as f:
                settings = json.load(f)
        except json.JSONDecodeError:
            settings = {"cSpell.words": []}

    current_words = set(settings.get("cSpell.words", []))
    new_words_count = 0
    for w in words:
        if w not in current_words:
            current_words.add(w)
            new_words_count += 1
            print(f"Added '{w}' to dictionary.")

    if new_words_count > 0:
        settings["cSpell.words"] = sorted(list(current_words))
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=4)
        print(f"✅ Added {new_words_count} words to .vscode/settings.json")
    else:
        print("No new words to add.")


BIO_TERMS = [
    "hyperbolic",
    "poincare",
    "embedding",
    "bioinformatics",
    "ternary",
    "vaes",
    "vae",
    "gcn",
    "cnn",
    "autoencoder",
    "decoder",
    "encoder",
    "latent",
    "manifold",
    "phylogeny",
    "phylogenetic",
    "genomic",
    "codon",
    "adjoint",
    "laplacian",
    "eigenvectors",
    "eigendecomposition",
    "tqdm",
    "argparse",
    "numpy",
    "matplotlib",
    "pyplot",
    "scikit",
    "sklearn",
    "pandas",
    "pytorch",
    "torch",
    "cuda",
    "cpu",
    "gpu",
]


def main():
    print("=== Codebase Maintenance Script ===")

    # 1. Format Code
    run_formatter()

    # 2. Add Common Terms to Dictionary
    print("\nUpdating Dictionary...")
    add_words_to_dictionary(BIO_TERMS)

    # 3. Future Linting (Optional)
    # run_linter()

    print("\n=== Maintenance Complete ===")


if __name__ == "__main__":
    main()
