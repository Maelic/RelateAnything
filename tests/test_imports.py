"""Import tripwire.

The single highest-value test in the suite: the repo shipped for weeks with
data/ (a source package train.py imports) swallowed by .gitignore, so every
fresh clone died at import while every local checkout worked. CI runs on a
fresh clone, so this file makes that class of bug impossible to reintroduce.
"""


def test_core_packages_import():
    import relsgg.api          # noqa: F401
    import relsgg.model        # noqa: F401
    import relsgg.evaluator    # noqa: F401
    import relsgg.geometry     # noqa: F401
    import relsgg.vocab        # noqa: F401


def test_data_package_imports():
    # THE regression this suite exists for.
    import data.relation_dataset   # noqa: F401
    import data.multipack          # noqa: F401


def test_deploy_modules_import():
    # deploy/ is a script dir (no __init__); conftest puts it on sys.path the
    # way the demo does. These must import WITHOUT torch being exercised —
    # the laptop runtime is numpy + onnxruntime only.
    import postprocess             # noqa: F401
    import vocab                   # noqa: F401


def test_train_entrypoint_imports():
    import train                   # noqa: F401
