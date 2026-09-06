"""Import tripwire.

The repository once shipped with ``data/`` — a source package the trainer
imports — excluded by.gitignore, so every fresh clone failed at import while
every local checkout worked. CI runs on a fresh clone, so this file makes
that class of failure impossible to reintroduce.
"""


def test_package_imports():
    import relsgg                       # noqa: F401
    import relsgg.api                   # noqa: F401
    import relsgg.config                # noqa: F401
    import relsgg.model                 # noqa: F401
    import relsgg.model.geometry        # noqa: F401
    import relsgg.data.dataset          # noqa: F401
    import relsgg.data.multipack        # noqa: F401
    import relsgg.eval.evaluator        # noqa: F401
    import relsgg.training.losses       # noqa: F401
    import relsgg.training.engine       # noqa: F401
    import relsgg.text.student          # noqa: F401


def test_torch_free_modules_import_without_torch():
    """The ONNX runtime ships without torch, and imports these two."""
    import relsgg.decompose              # noqa: F401
    import relsgg.scoring                # noqa: F401


def test_deploy_modules_import():
    # deploy/ is a script directory (no __init__.py — its files run directly
    # on edge devices); conftest puts it on sys.path as the demo does.
    import postprocess                   # noqa: F401
    import vocab                         # noqa: F401


def test_train_entrypoint_imports():
    import train                         # noqa: F401
