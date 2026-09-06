"""Shared test setup.

deploy/ is a script directory, not a package (no __init__.py — its files are
run directly on edge devices with no install step). Put it on sys.path so
tests import its modules the same way the demo does.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_ROOT, os.path.join(_ROOT, "deploy")):
    if p not in sys.path:
        sys.path.insert(0, p)
