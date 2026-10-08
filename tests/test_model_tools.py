from pathlib import Path

import pytest

from osai.errors import ConfigurationError
from osai.model_tools import find_model_bundle


@pytest.mark.parametrize(
    "relative",
    ["outputs/gguf", "outputs/merged-model/gguf", "outputs/checkpoint/merged-model/gguf"],
)
def test_model_tools_finds_current_and_older_session_bundles(tmp_path: Path, relative: str):
    bundle = tmp_path / relative
    bundle.mkdir(parents=True)
    (bundle / "osai_fusion.json").write_text("{}", encoding="utf-8")
    assert find_model_bundle(tmp_path) == bundle
    assert find_model_bundle(tmp_path / "outputs") == bundle
    assert find_model_bundle(bundle) == bundle


def test_model_tools_rejects_unrelated_folder(tmp_path: Path):
    with pytest.raises(ConfigurationError, match="model folder"):
        find_model_bundle(tmp_path)
