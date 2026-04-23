# fast.ai Course Practice

This repository is a personal learning and practice workspace for the
[fast.ai Practical Deep Learning for Coders](https://course.fast.ai/) course,
covering both Part 1 and Part 2.

The main focus is Part 2, "From Deep Learning Foundations to Stable Diffusion",
where the course builds toward diffusion models by implementing many deep
learning foundations from scratch. Part 1 is also represented and is almost
complete in this repo.

## Course Focus

Part 1 introduces practical deep learning workflows using notebooks and a
top-down teaching style. The notebooks in this repository cover core applied
topics such as image classification, neural network foundations, NLP, model
iteration, and collaborative filtering.

Part 2 goes deeper into the foundations behind modern deep learning systems.
According to the official fast.ai Part 2 overview, the course implements Stable
Diffusion from scratch and covers topics including DDPM, DDIM, samplers,
autoencoders, U-Nets, ResNets, transformers, CLIP embeddings, initialization,
normalization, accelerated training, mixed precision, and experiment tracking.

During Part 2, the course incrementally develops a small deep learning
framework/app similar in spirit to `miniai`. In this repository, that evolving
source code lives in `src/` and is used by the later Part 2 notebooks.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `part1_nbs/` | Part 1 notebooks for practical deep learning exercises and experiments. |
| `part2_nbs/` | Part 2 notebooks, with the primary focus on deep learning foundations and diffusion models. |
| `part2_nbs/00_stable_diffusion_intro/` | Introductory Stable Diffusion notebooks that inspect and use Hugging Face Diffusers pipelines. |
| `src/` | Incrementally developed Part 2 source code used by the notebooks. |
| `.github/workflows/sync-src-to-drive.yml` | GitHub Action that syncs `src/` to Google Drive for Colab usage. |
| `pyproject.toml` | Python project metadata and dependencies. |
| `uv.lock` | Locked dependency resolution managed by `uv`. |

## Environment Setup

Python dependencies are managed with [`uv`](https://docs.astral.sh/uv/). Use
`uv` to recreate the local environment rather than installing packages manually.

```bash
git clone <repo-url>
cd fast-ai
uv sync
```

The project currently targets Python `>=3.12` and includes dependencies used
throughout the notebooks, including PyTorch, torchvision, diffusers, datasets,
accelerate, timm, torcheval, matplotlib, ipywidgets, numba, and W&B.

To register the environment as a Jupyter kernel:

```bash
uv run python -m ipykernel install --user --name fast-ai --display-name "Python (fast-ai)"
```

If you do not already have a Jupyter frontend available, you can launch one
through `uv` without adding it permanently to the project:

```bash
uv run --with jupyterlab jupyter lab
```

Some diffusion notebooks benefit from a CUDA-enabled GPU. You can check the
active PyTorch device support with:

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```

## Working With Notebooks

Open the notebooks from the repository root so imports such as
`from src.utils import ...` resolve correctly.

Recommended order:

1. Use `part1_nbs/` for the Part 1 applied deep learning exercises.
2. Use `part2_nbs/` for the Part 2 foundations and diffusion work.
3. Treat `src/` as the shared source code that later Part 2 notebooks import
   and extend.

Several notebooks download external datasets or model weights from services
such as Hugging Face. For notebooks that require authenticated access, configure
the relevant tokens in your local environment or in the notebook platform you
are using.

## Colab and Google Drive Sync

Later Part 2 notebooks depend on source files developed in earlier notebooks.
To make those imports available in Google Colab, this repository includes a
GitHub Action that syncs the local `src/` directory to a fixed Google Drive
location.

The workflow in `.github/workflows/sync-src-to-drive.yml`:

- runs on pushes to `main` when files under `src/**` change;
- can also be started manually with `workflow_dispatch`;
- installs `rclone`;
- reads the `RCLONE_CONFIG` GitHub secret;
- performs a dry run; and
- syncs `./src` to the configured `gdrive_minisd:src` target.

In Colab, mount Google Drive and add the parent directory that contains `src/`
to `sys.path`. The notebooks are written to import modules as `src.*`, so the
path should point to the parent folder, not to `src/` itself.

```python
from google.colab import drive
drive.mount("/content/drive")

import sys
app_path = "/content/drive/MyDrive/Projects/miniSD"  # parent directory containing src/
if app_path not in sys.path:
    sys.path.append(app_path)
```

Adjust `app_path` if the `gdrive_minisd` rclone remote points to a different
Google Drive folder.

## Key References

- [fast.ai Practical Deep Learning for Coders](https://course.fast.ai/)
- [fast.ai Part 2 overview](https://course.fast.ai/Lessons/part2.html)
- [fastai/course22p2](https://github.com/fastai/course22p2)
- [uv documentation](https://docs.astral.sh/uv/)

## Acknowledgements

This project uses and adapts materials from the
[fast.ai](https://course.fast.ai/) course and notebooks. The original fast.ai
materials are licensed under the
[Apache License 2.0](https://github.com/fastai/fastbook/blob/master/LICENSE).

Some notebooks and code have been modified for learning and experimentation in
this repository. All credit for the original course materials goes to fast.ai
and its contributors.
