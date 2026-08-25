# How work lands in this repo

## The loop

1. You send Claude a task list, roughly weekly.
2. Claude creates a branch named `<type>/<short-description>`.
3. Claude commits the work and opens a pull request against `main`.
4. You review the PR, comment or request changes, and merge when you are happy.

Nothing is ever pushed directly to `main`. Every change arrives as a PR you control.

## Branch naming

| Prefix | Use for |
|---|---|
| `setup/` | Project structure, tooling, config |
| `data/` | Loading, preprocessing, caching, folds |
| `labels/` | Report parsing and pseudo-label generation |
| `model/` | Architectures and training code |
| `exp/` | A specific experiment run |
| `docs/` | Notes, plans, writeups |
| `fix/` | Corrections to existing code |

## What a good PR from Claude contains

- A description of what changed and why
- What was tested and what was not
- Any assumption Claude had to make that you should check
- Open questions, if any

## Reviewing when you are new

You do not need to verify every line. Focus on:

- **Does it match what you asked for?** If not, say so in the PR.
- **Do you understand what it does?** If a file is opaque, ask for comments or a
  walkthrough. Merging code you do not understand builds a project you cannot debug
  in week seven.
- **Does it touch the CV split?** After Week 2 the fold assignment must not change.
  Treat any change to `data/folds.csv` or the splitting logic as a red flag.

## Experiment log

`docs/experiments.md` is append only. Every training run gets a row. Claude adds the
row in the same PR as the run.

## What Claude cannot do

- Run training on the competition data. There is no GPU and no dataset access in
  Claude's environment. Claude writes and reviews the code; you run it on Kaggle or
  locally and report results back.
- Submit to Kaggle on your behalf.
- Merge its own PRs.

So the practical division: Claude does the reading, planning, code writing, debugging
and documentation. You do the running, submitting and deciding.
