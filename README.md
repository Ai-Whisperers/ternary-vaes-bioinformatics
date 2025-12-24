# Ternary VAE Bioinformatics

[![License: PolyForm Non‑Commercial 1.0.0](https://img.shields.io/badge/License-PolyForm%20Non‑Commercial%201.0.0-lightgrey.svg)](LICENSE)
[![License: CC‑BY‑4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](RESULTS_LICENSE.md)
[![Open Medicine Policy](https://img.shields.io/badge/Open%20Medicine-Policy-blue.svg)](OPEN_MEDICINE_POLICY.md)
[![GitHub stars](https://img.shields.io/github/stars/Alejandro/ternary-vaes-bioinformatics?style=social)](https://github.com/Alejandro/ternary-vaes-bioinformatics)

---

## 📖 Overview

**Ternary VAE** is a cutting‑edge variational auto‑encoder that learns representations in **hyperbolic geometry** and **3‑adic number spaces**. It is designed for bioinformatics applications such as:

- **Geometric vaccine design** for HIV and emerging pathogens.
- **Drug‑interaction modeling** using manifold‑centric embeddings.
- **Codon‑space exploration** for synthetic biology.
- **Agricultural drug discovery** (e.g., Pasteur Molecule‑Binding project).

The project follows an **Open‑Medicine** philosophy: all scientific outputs (data, figures, model weights) are released under **CC‑BY‑4.0**, while the source code remains under the **PolyForm Non‑Commercial 1.0.0** license to prevent exclusive commercial exploitation.

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/Alejandro/ternary-vaes-bioinformatics.git
cd ternary-vaes-bioinformatics

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the core training script (example)
python scripts/train_vae.py --config configs/default.yaml
```

> **Tip**: The repository includes a `Dockerfile` for reproducible container builds.

---

## 📚 Documentation

- **Theory & Foundations** – detailed mathematical background, biological context, and validation strategy: `DOCUMENTATION/01_PROJECT_KNOWLEDGE_BASE/02_THEORY_AND_FOUNDATIONS/`
- **Research Proposals** – organized proposals for future work: `DOCUMENTATION/01_PROJECT_KNOWLEDGE_BASE/02_THEORY_AND_FOUNDATIONS/09_BIBLIOGRAPHY_AND_RESOURCES/RESEARCH_PROPOSALS/`
- **Project Management** – roadmaps, risk register, and code‑health metrics: `DOCUMENTATION/02_PROJECT_MANAGEMENT/`
- **API Reference** – generated automatically from the `src/` package (see `docs/` after running `scripts/doc_builder.py`).

---

## 🛠️ Installation & Development

The project uses a standard Python stack. For development, we recommend the following tools:

- **ruff** – fast Python linter/formatter (`ruff.toml` is already configured).
- **pytest** – test suite (`tests/`).
- **pre‑commit** – hooks for linting and SPDX header checks.
- **GitHub Actions** – CI pipelines for linting, testing, and code‑health dashboards.

To set up pre‑commit:

```bash
pip install pre-commit
pre-commit install
```

---

## 📦 License & Legal

### Software (Code)

- **License**: PolyForm Non‑Commercial 1.0.0
- **Permitted**: Academic, educational, and non‑profit use.
- **Commercial Use**: Requires a separate commercial license – contact `support@aiwhisperers.com`.

### Research Outputs (Data, Figures, Models)

- **License**: CC‑BY‑4.0 – free for any reuse with attribution.
- **Open‑Medicine Policy**: See `OPEN_MEDICINE_POLICY.md` for detailed terms.

All legal documents are collected in the **Legal & IP** directory:

- `LICENSE`
- `NOTICE`
- `OPEN_MEDICINE_POLICY.md`
- `RESULTS_LICENSE.md`
- `CLA.md`
- `CONTRIBUTING.md`
- `CODE_OF_CONDUCT.md`
- `SECURITY.md`
- `CITATION.cff`

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Read the Contributor License Agreement** (`CLA.md`) and sign it.
2. **Review the Code of Conduct** (`CODE_OF_CONDUCT.md`).
3. **Check the contribution guidelines** (`CONTRIBUTING.md`) for branch naming, commit style, and testing requirements.
4. **Open a Pull Request** – link it to an existing issue or create a new one.
5. **Ensure all CI checks pass** (ruff, pytest, SPDX header verification).

---

## 🔒 Security

Report any vulnerabilities via the `SECURITY.md` policy. We use a coordinated disclosure process and will acknowledge contributors.

---

## 📑 Citation

Please cite the project using the provided `CITATION.cff`. Example BibTeX entry:

```bibtex
@software{ternary_vae,
  author = {Alejandro, et al.},
  title = {Ternary VAE Bioinformatics},
  year = {2025},
  url = {https://github.com/Alejandro/ternary-vaes-bioinformatics},
  doi = {10.5281/zenodo.XXXXXX}
}
```

---

## 👥 Authors & Acknowledgments

- **Primary Authors** – see `AUTHORS.md`.
- **Contributors** – see `CONTRIBUTORS.md`.
- **Funding** – this work is supported by open‑science grants and institutional collaborations.

---

## 📞 Contact

For general questions, open an issue. For commercial licensing inquiries, email `support@aiwhisperers.com`.

---

_Last updated: 2025‑12‑24_
