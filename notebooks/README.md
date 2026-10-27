# Notebooks

`walkthrough.ipynb` walks through every experiment in the order of the talk
and the paper: the data, E1 (disk simulation), E2 (GPU-trained predictors),
E3 (real device), E4 and E5 (web replay), E6 (live HTTP), every
pre-registered decision, and the classifier studies added later (E7 and
after, shown once their `results/` directories exist). Every number in it is
computed by a cell from the tracked `results/` tree. The markdown cells only
say which results file and which paper section or table a cell belongs to.

The notebook is committed **with its outputs**, so you can read it without
running anything: open it in VS Code, or view it on GitHub.

## Opening and running it in VS Code

1. Install the **Python** and **Jupyter** extensions (both from Microsoft).
2. Open the repository folder and then `notebooks/walkthrough.ipynb`.
3. Use **Select Kernel** (top right), then **Python Environments**, and pick
   the repository's venv, `.venv/bin/python`. The venv needs the `[web,paper]`
   extras (pandas, numpy, matplotlib, scipy). The first time you run a cell,
   VS Code offers to install `ipykernel` into the venv. Accept it, because
   the venv does not ship with one.
4. Choose **Run All**. You do not need `PYTHONPATH`: the first cell finds
   the repository root, changes into it and puts `src/` on `sys.path`
   itself. It loads `paper/genlib.py` and `paper/decisions.py` by file path,
   because `paper/` must never be on `sys.path` (`paper/numbers.py` would
   shadow the standard library's `numbers`).

## Rebuilding it after new results land

`build_notebook.py` defines the cells. It also executes them in order
without a Jupyter kernel (Agg backend, cwd = the repository root) and stores
their outputs, so the committed file opens pre-rendered:

```bash
PYTHONPATH=src .venv/bin/python reproduce.py paper      # first, if results/ changed
.venv/bin/python reproduce.py notebook                  # rebuilds notebooks/walkthrough.ipynb
```

The build is deterministic: fixed cell ids, no timestamps, PNGs at a fixed
DPI with their metadata stripped. `tests/test_notebook.py` fails when the
committed notebook differs from a fresh build, so commit the rebuilt file
together with any change to `results/`. To change the notebook, edit
`CELLS` in `build_notebook.py` and rebuild; do not edit the `.ipynb` by hand.
