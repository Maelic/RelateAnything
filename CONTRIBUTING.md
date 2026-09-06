# Contributing

## Setup

```bash
./install.sh
pip install -e ".[dev]"
pytest tests/ -q
```

The test suite is CPU-only, needs no weights and no network, and is what CI
runs. It covers the install path, imports, evaluator math, score parity between
the training and deploy paths, and edge decoding. Model-forward parity lives in
local scripts because it needs weights.

## Repository conventions

**Numbers are generated, never typed.** Model cards come from
`release/make_model_cards.py`, and the paper's tables by the same rule; both
reading measured eval JSONs. If you find yourself editing a number in a
markdown file or a `.tex` table, you are editing the wrong file.

**Every flag defaults to the previous behaviour.** New configuration fields are
written so that the default reproduces prior runs bit-identically, and the
docstring says so. This is what makes an ablation a single-variable experiment
rather than a hope.

**Document the measurement, not the intention.** The comments in
`RelSGGConfig` and in `relsgg/scoring.py` are the model of what a comment should
be here: what was measured, what it cost, and what the alternative did. A flag
whose comment says "improves quality" is not documented.

**Synonyms are never collapsed** in emitted data. Canonical groups exist for
loss computation only.

**Negative results stay in the repository.** `docs/pitfalls.md`, the SPEC's
rationale sections and the manifest's `resolution` fields record what did not
work and why. Deleting them re-opens closed questions.

## Adding a configuration field

1. Add it to `RelSGGConfig` in [`relsgg/config.py`](relsgg/config.py). Its
   default is what ships, so a new field's default must be the released
   behaviour — `tests/test_config.py` checks that every released model differs
   from `RelSGGConfig()` only in its backbone.
2. Say inline what it does and why it exists.
3. Add the argparse flag in `train.py` with the same name, and it round-trips
   into and out of a checkpoint by itself: `config_from_args` reads every field
   by its own name.

## Adding a dataset

See [data.md](docs/data.md). The leakage audit is not optional.

## Running experiments

See [training.md](docs/training.md). Use the 50K proxy pack for ablations:
about 2.5 GPU-hours per arm instead of 24, once you have checked that the
effect you are chasing shows up on it at all.

## Style

`black` and `isort` are in the `dev` extra. There is no enforcement hook; match
the surrounding file.

## Reporting a number

Read [evaluation.md](docs/evaluation.md) and [pitfalls.md](docs/pitfalls.md) first. In
particular: graph-constrained, final epoch, and never from the training log.
