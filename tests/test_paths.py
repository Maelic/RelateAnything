"""Image folders of packs built on another machine are found under $RA_DATASETS."""
import json
import warnings

import pytest

from relsgg.paths import relocated, resolve_img_dir


def test_relocated_keeps_the_part_after_the_last_datasets_component():
    assert relocated("/mimer/x/DATASETS/MEGASG/train", "/data") == "/data/MEGASG/train"
    assert relocated("/a/datasets/b/datasets/PSG/test", "/r") == "/r/PSG/test"
    with pytest.raises(ValueError):
        relocated("/no/anchor/here", "/r")


def test_recorded_folder_is_used_when_it_exists(tmp_path, monkeypatch):
    monkeypatch.setenv("RA_DATASETS", str(tmp_path / "elsewhere"))
    assert resolve_img_dir({"img_dir": str(tmp_path)}) == tmp_path


def test_falls_back_to_ra_datasets(tmp_path, monkeypatch):
    (tmp_path / "VG150_coco_format" / "test").mkdir(parents=True)
    monkeypatch.setenv("RA_DATASETS", str(tmp_path))
    meta = {"img_dir": "/other/machine/datasets/VG150_coco_format/test"}
    assert resolve_img_dir(meta) == tmp_path / "VG150_coco_format" / "test"
    # a pack already rewritten by setup keeps the original path in img_dir_built
    meta = {"img_dir": "/stale/root/VG150_coco_format/test",
            "img_dir_built": "/mimer/DATASETS/VG150_coco_format/test"}
    assert resolve_img_dir(meta) == tmp_path / "VG150_coco_format" / "test"


def test_warns_with_every_path_tried(tmp_path, monkeypatch):
    monkeypatch.setenv("RA_DATASETS", str(tmp_path))
    meta = {"img_dir": "/other/datasets/PSG_coco_format/test"}
    with pytest.warns(UserWarning) as rec:
        out = resolve_img_dir(meta)
    msg = str(rec[0].message)
    assert "/other/datasets/PSG_coco_format/test" in msg and str(tmp_path / "PSG_coco_format") in msg
    assert str(out) == meta["img_dir"]
