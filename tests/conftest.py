"""
conftest.py
-----------
Shared test fixtures for CrossVerify_M3.
Generates small synthetic test fixtures on the fly so tests don't depend
on external dataset images or check binary assets into git.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from PIL import Image, ImageDraw


def make_text_image(tmp_path: Path, lines, filename="text.png") -> str:
    """Renders a list of text lines onto a plain white synthetic image."""
    img = Image.new("RGB", (500, 300), color="white")
    draw = ImageDraw.Draw(img)
    y = 10
    for line in lines:
        draw.text((10, y), line, fill="black")
        y += 30
    path = tmp_path / filename
    img.save(path)
    return str(path)


def make_blank_image(tmp_path: Path, filename="blank.png") -> str:
    img = Image.new("RGB", (200, 200), color="white")
    path = tmp_path / filename
    img.save(path)
    return str(path)


def make_qr_image(tmp_path: Path, payload: str, filename="qr.png") -> str:
    import qrcode

    img = qrcode.make(payload).convert("RGB")
    path = tmp_path / filename
    img.save(path)
    return str(path)


@pytest.fixture
def text_image_factory(tmp_path):
    return lambda lines, filename="text.png": make_text_image(tmp_path, lines, filename)


@pytest.fixture
def blank_image(tmp_path):
    return make_blank_image(tmp_path)


@pytest.fixture
def qr_image_factory(tmp_path):
    return lambda payload, filename="qr.png": make_qr_image(tmp_path, payload, filename)
