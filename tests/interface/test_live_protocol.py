import json
import pathlib

import jsonschema

from core.entities.pose_entity import Keypoint, KeypointCollection
from core.service.pose_feature_service import PoseFeatureService
from interface.ws.live_protocol import build_bands_message, build_coaching_message

_SCHEMAS = pathlib.Path(__file__).resolve().parents[2] / "contract" / "schemas"


def _schema(name):
    return json.loads((_SCHEMAS / f"{name}.schema.json").read_text(encoding="utf-8"))


_BENCH_MIXED = {
    "left_shoulder_left_elbow_left_wrist": 90.0,  # green
    "right_shoulder_right_elbow_right_wrist": 130.0,  # red
}


class TestBuildBandsMessage:
    def test_conforms_to_contract_and_keys_by_vertex(self):
        msg = build_bands_message(_BENCH_MIXED, "bench_press", t=1719900000123)
        jsonschema.validate(msg, _schema("bands"))
        assert msg["type"] == "bands"
        assert msg["t"] == 1719900000123
        assert msg["bands"]["left_elbow"] == "green"
        assert msg["bands"]["right_elbow"] == "red"

    def test_missing_angles_are_not_banded(self):
        msg = build_bands_message({}, "bench_press", t=1)
        assert msg["bands"] == {}


class TestBuildCoachingMessage:
    def test_conforms_to_contract_with_cue(self):
        bad = {
            "left_shoulder_left_elbow_left_wrist": 130.0,
            "right_shoulder_right_elbow_right_wrist": 130.0,
        }
        msg = build_coaching_message(bad, "bench_press")
        jsonschema.validate(msg, _schema("coaching"))
        assert msg["type"] == "coaching"
        assert msg["accuracy"] < 1.0
        assert any("elbow" in s.lower() for s in msg["suggestions"])

    def test_good_form_conforms_and_scores_full(self):
        good = {
            "left_shoulder_left_elbow_left_wrist": 90.0,
            "right_shoulder_right_elbow_right_wrist": 90.0,
        }
        msg = build_coaching_message(good, "bench_press")
        jsonschema.validate(msg, _schema("coaching"))
        assert msg["accuracy"] == 1.0


class TestKeypointsInEndToEnd:
    """Deterministic keypoints-in scoring boundary (ADR 0013): a client keypoint
    frame -> features -> contract-conforming bands, with no server ML."""

    def _kc(self, **joints):
        kps = {
            name: Keypoint(x=float(v[0]), y=float(v[1]), confidence=1.0)
            for name, v in joints.items()
        }
        return KeypointCollection(keypoints=kps)

    def test_keypoints_frame_produces_expected_bands(self):
        # Left arm straight (elbow ~180 deg -> red); right arm ~90 deg (green).
        keypoints = self._kc(
            left_shoulder=(0.4, 0.3),
            left_elbow=(0.4, 0.5),
            left_wrist=(0.4, 0.7),
            right_shoulder=(0.6, 0.3),
            right_elbow=(0.6, 0.5),
            right_wrist=(0.8, 0.5),
        )
        features = PoseFeatureService().process_features(keypoints)
        msg = build_bands_message(features.joint_angles, "bench_press", t=42)
        jsonschema.validate(msg, _schema("bands"))
        assert msg["bands"]["left_elbow"] == "red"
        assert msg["bands"]["right_elbow"] == "green"
