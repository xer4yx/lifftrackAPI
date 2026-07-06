import copy
import glob
import hashlib
import json
import os
import pathlib

import jsonschema
import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_CONTRACT = _ROOT / "contract"
_SCHEMAS = _CONTRACT / "schemas"
_EXAMPLES = _CONTRACT / "examples"

# Drift guard: canonical-JSON checksum of the schema files, shared verbatim with
# the app repo. If a schema changes, bump PROTOCOL_VERSION and update this value
# (the test prints the actual on failure), then mirror contract/ to the app.
EXPECTED_CONTRACT_CHECKSUM = (
    "92fa484b85b6e53402a1bd285be0f280b25e7083e3c0dc1b945f665ccc80f917"
)


def _load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _canonical_checksum():
    h = hashlib.sha256()
    for f in sorted(glob.glob(str(_SCHEMAS / "*.schema.json"))):
        canon = json.dumps(_load(f), sort_keys=True, separators=(",", ":"))
        h.update(os.path.basename(f).encode())
        h.update(b"\0")
        h.update(canon.encode())
        h.update(b"\0")
    return h.hexdigest()


_MESSAGE_TYPES = ["keypoints", "bands", "coaching"]


@pytest.mark.parametrize("message_type", _MESSAGE_TYPES)
def test_golden_example_validates_against_schema(message_type):
    schema = _load(_SCHEMAS / f"{message_type}.schema.json")
    example = _load(_EXAMPLES / f"{message_type}.example.json")
    jsonschema.validate(example, schema)  # raises on non-conformance


def test_keypoints_reject_out_of_range_coordinate():
    schema = _load(_SCHEMAS / "keypoints.schema.json")
    bad = _load(_EXAMPLES / "keypoints.example.json")
    bad["keypoints"]["nose"]["x"] = 1.5  # outside [0, 1]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)


def test_keypoints_reject_unknown_joint_name():
    schema = _load(_SCHEMAS / "keypoints.schema.json")
    bad = _load(_EXAMPLES / "keypoints.example.json")
    bad["keypoints"]["third_arm"] = {"x": 0.5, "y": 0.5, "confidence": 0.9}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)


def test_bands_reject_invalid_color():
    schema = _load(_SCHEMAS / "bands.schema.json")
    bad = _load(_EXAMPLES / "bands.example.json")
    bad["bands"]["left_elbow"] = "yellow"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)


def test_coaching_reject_accuracy_above_one():
    schema = _load(_SCHEMAS / "coaching.schema.json")
    bad = _load(_EXAMPLES / "coaching.example.json")
    bad["accuracy"] = 1.5
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)


def test_protocol_version_is_present():
    version = (_CONTRACT / "PROTOCOL_VERSION").read_text(encoding="utf-8").strip()
    assert version, "PROTOCOL_VERSION is empty"
    assert len(version.split(".")) == 3, f"expected semantic version, got {version!r}"


def test_contract_checksum_matches_expected():
    actual = _canonical_checksum()
    assert actual == EXPECTED_CONTRACT_CHECKSUM, (
        "Contract schemas changed. If intentional: bump PROTOCOL_VERSION, set "
        f"EXPECTED_CONTRACT_CHECKSUM = {actual!r}, and mirror contract/ to the app repo."
    )
