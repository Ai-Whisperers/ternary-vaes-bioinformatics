# Gemini Agent Context

**Doc-Type:** Agent Configuration · Version 1.0

---

## Core Mandates

### 1. Environment Isolation
- **ALWAYS** operate within an isolated virtual environment (`.venv`).
- Verify activation before running Python scripts.
- Do not rely on system-wide packages.

### 2. Cache Management
- **STRICTLY** save machine learning caches (HuggingFace, Torch, etc.) within the project directory.
- Use `.cache/` (ensure it is gitignored) or a specified local directory.
- **NEVER** save caches to the system root, user home (outside project), or global system paths.
- Set environment variables if necessary (e.g., `HF_HOME`, `TORCH_HOME`) to point to project-local paths during execution.

---
