"""Publish a group of prepared model files without overwriting existing assets."""

import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory


@contextmanager
def staged_assets(destination, expected):
    destination.mkdir(exist_ok=True, parents=True)
    for name in expected:
        if (destination / name).exists():
            raise FileExistsError(destination / name)
    published = []
    with TemporaryDirectory(dir=destination) as temporary:
        stage = Path(temporary)
        yield stage
        try:
            for name in expected:
                target = destination / name
                os.link(stage / name, target)
                published.append(target)
        except OSError:
            for target in published:
                target.unlink()
            raise
