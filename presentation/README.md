# Presentation

| File | What |
|---|---|
| `slides.tex` | the Beamer deck: 15 slides, one per section of [`docs/PRESENTATION.md`](../docs/PRESENTATION.md), with `\note{}` speaker notes |
| [`docs/PRESENTATION.md`](../docs/PRESENTATION.md) | the same talk in full: what each slide shows, what to say, the demo, and the viva questions with answers |

The deck has no numbers or figures of its own. Every number is a macro from
`paper/numbers.tex` and every figure is a PDF from `paper/figures/`, both
written by `python reproduce.py paper` from `results/`. Regenerate them
first, then lint:

```bash
PYTHONPATH=src .venv/bin/python reproduce.py paper   # numbers.tex, figures, tables
python3 tools/check_doc_numbers.py                   # also fails on a hand-typed figure in slides.tex
```

## Compiling on Overleaf

No LaTeX toolchain is installed on the development machine (lead decision
L6), so the deck, like the paper, is compiled on Overleaf.

1. Create a blank Overleaf project.
2. Upload two folders from the repository, keeping their names:
   - `paper/` with at least `numbers.tex` and `figures/*.pdf`;
   - `presentation/` with `slides.tex`.

   The project then has `paper/numbers.tex`, `paper/figures/` and
   `presentation/slides.tex`. (Uploading the whole `paper/` folder is fine;
   it is also what the paper itself needs.)
3. Menu, Settings: Compiler **pdfLaTeX**; Main document
   **presentation/slides.tex**.
4. Recompile.

`slides.tex` looks for `../paper/numbers.tex` and `../paper/figures/` (the
repository layout) and falls back to `paper/numbers.tex` and
`paper/figures/`, so it compiles whether Overleaf runs from the project
root or from the `presentation/` folder. Only standard packages are used
(`beamer`, `graphicx`, `fontenc`), all in Overleaf's TeX Live.

- To print the speaker notes after each slide, uncomment
  `\setbeameroption{show notes}` near the top of `slides.tex`.
- A bold **??** on a slide means `numbers.tex` has no macro of that name:
  either `reproduce.py paper` has not been run since `results/` changed, or
  that results row does not exist. Do not type a number in its place.
- The figures before `reproduce.py paper` runs are placeholders stamped
  "MISSING"; after it they are drawn from `results/`.

## What is not here

No `.pptx` is generated: `python-pptx` is not available in this
environment (plan C14). The two Review 2 `.pptx` decks were removed from
`master` because their figures were out of date; they remain in git
history, last present at commit `66226bb`.
