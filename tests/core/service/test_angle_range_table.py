import pytest

from core.entities.pose_entity import Keypoint, KeypointCollection
from core.service.angle_range_table import (
    AngleRange,
    Band,
    EXERCISE_ANGLE_RANGES,
    assess_angles,
    classify_band,
    get_angle_ranges,
    score_form,
)
from core.service.pose_feature_service import PoseFeatureService

# Exercise-name variants the form-analysis pipeline recognizes (kept in sync with
# tests/interface/test_exercise_thresholds.py and FormAnalysisService).
RECOGNIZED_EXERCISES = [
    "bench_press",
    "benchpress",
    "deadlift",
    "romanian_deadlift",
    "rdl",
    "shoulder_press",
    "overhead_press",
]

_COCO_KEYPOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def _emittable_angle_keys():
    """Every joint-angle key the pipeline can produce from a full COCO-17 pose."""
    kps = {
        name: Keypoint(x=0.5, y=0.5 + 0.01 * i, confidence=1.0)
        for i, name in enumerate(_COCO_KEYPOINTS)
    }
    angles = PoseFeatureService().extract_joint_angles(
        KeypointCollection(keypoints=kps)
    )
    return set(angles.keys())


def _range(lo=70.0, hi=110.0, tol=15.0):
    return AngleRange(
        joint_angle="left_shoulder_left_elbow_left_wrist",
        vertex="left_elbow",
        lo=lo,
        hi=hi,
        tolerance=tol,
        cue="Keep your elbow around 90 degrees.",
    )


class TestClassifyBand:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (90.0, Band.GREEN),  # inside the range
            (70.0, Band.GREEN),  # lower edge is inclusive
            (110.0, Band.GREEN),  # upper edge is inclusive
            (65.0, Band.ORANGE),  # within tolerance below the range
            (120.0, Band.ORANGE),  # within tolerance above the range
            (55.0, Band.ORANGE),  # exactly lo - tolerance
            (125.0, Band.ORANGE),  # exactly hi + tolerance
            (54.0, Band.RED),  # beyond tolerance below
            (126.0, Band.RED),  # beyond tolerance above
        ],
    )
    def test_bands(self, value, expected):
        assert classify_band(value, _range()) == expected


class TestTableData:
    def test_every_recognized_exercise_has_ranges(self):
        missing = [e for e in RECOGNIZED_EXERCISES if not get_angle_ranges(e)]
        assert missing == [], f"exercises with no angle ranges: {missing}"

    def test_aliases_resolve_to_canonical_ranges(self):
        assert get_angle_ranges("benchpress") == get_angle_ranges("bench_press")
        assert get_angle_ranges("romanian_deadlift") == get_angle_ranges("rdl")
        assert get_angle_ranges("overhead_press") == get_angle_ranges("shoulder_press")

    def test_ranges_are_well_formed(self):
        for exercise, ranges in EXERCISE_ANGLE_RANGES.items():
            assert ranges, f"{exercise} has no ranges"
            for r in ranges:
                assert r.lo < r.hi, f"{exercise}/{r.joint_angle}: lo >= hi"
                assert r.tolerance > 0, f"{exercise}/{r.joint_angle}: tolerance <= 0"
                assert r.cue.strip(), f"{exercise}/{r.joint_angle}: empty cue"

    def test_vertex_is_the_middle_keypoint_of_the_angle(self):
        # The band is colored at the joint's vertex, which must be the middle
        # keypoint of the "a_b_c" angle key.
        for exercise, ranges in EXERCISE_ANGLE_RANGES.items():
            for r in ranges:
                assert r.vertex in r.joint_angle, (
                    f"{exercise}/{r.joint_angle}: vertex {r.vertex} not in angle key"
                )

    def test_table_angle_keys_are_emitted_by_the_pipeline(self):
        # Vocabulary-closure (ADR 0013): every angle the table scores must be one
        # the pipeline can actually produce, or it silently never bands.
        emittable = _emittable_angle_keys()
        unknown = sorted(
            {
                r.joint_angle
                for ranges in EXERCISE_ANGLE_RANGES.values()
                for r in ranges
                if r.joint_angle not in emittable
            }
        )
        assert unknown == [], f"table references angles the pipeline never emits: {unknown}"


class TestAssessAngles:
    def test_bands_present_angles_at_their_vertex(self):
        joint_angles = {
            "left_shoulder_left_elbow_left_wrist": 90.0,  # green
            "right_shoulder_right_elbow_right_wrist": 130.0,  # red (past 110+15)
        }
        by_vertex = {a.vertex: a for a in assess_angles(joint_angles, "bench_press")}
        assert by_vertex["left_elbow"].band == Band.GREEN
        assert by_vertex["left_elbow"].cue is None
        assert by_vertex["right_elbow"].band == Band.RED
        assert by_vertex["right_elbow"].cue

    def test_missing_angles_are_skipped(self):
        assessments = assess_angles(
            {"right_shoulder_right_elbow_right_wrist": 90.0}, "bench_press"
        )
        assert [a.vertex for a in assessments] == ["right_elbow"]


class TestScoreForm:
    _PERFECT = {
        "left_shoulder_left_elbow_left_wrist": 90.0,
        "right_shoulder_right_elbow_right_wrist": 90.0,
    }
    _BAD = {
        "left_shoulder_left_elbow_left_wrist": 130.0,
        "right_shoulder_right_elbow_right_wrist": 130.0,
    }

    def test_perfect_form_full_score_no_suggestions(self):
        accuracy, suggestions = score_form(self._PERFECT, "bench_press")
        assert accuracy == pytest.approx(1.0)
        assert suggestions == []

    def test_worse_form_scores_lower_with_cue(self):
        good, _ = score_form(self._PERFECT, "bench_press")
        bad, cues = score_form(self._BAD, "bench_press")
        assert bad < good
        assert cues

    def test_score_is_not_a_degenerate_constant(self):
        scores = {
            score_form(
                {
                    "left_shoulder_left_elbow_left_wrist": v,
                    "right_shoulder_right_elbow_right_wrist": v,
                },
                "bench_press",
            )[0]
            for v in (90.0, 118.0, 130.0)  # green, orange, red
        }
        assert len(scores) > 1
        assert scores != {0.0}
        assert scores != {1.0}


class TestShoulderPressDiscrimination:
    """P3 / ADR 0013: a real overhead press must out-score an arm-raise instead
    of everything reading ~100%. Driven through the real keypoint -> angle
    pipeline so the table and PoseFeatureService agree on the angle key."""

    @staticmethod
    def _score(shoulders, hips, wrists):
        # Synthetic COCO pose (normalized [0,1], y-down per ADR 0003) with just
        # the keypoints the wrist-shoulder-hip angle needs.
        (lsx, lsy), (rsx, rsy) = shoulders
        (lhx, lhy), (rhx, rhy) = hips
        (lwx, lwy), (rwx, rwy) = wrists
        kps = {
            "left_shoulder": Keypoint(x=lsx, y=lsy, confidence=1.0),
            "right_shoulder": Keypoint(x=rsx, y=rsy, confidence=1.0),
            "left_hip": Keypoint(x=lhx, y=lhy, confidence=1.0),
            "right_hip": Keypoint(x=rhx, y=rhy, confidence=1.0),
            "left_wrist": Keypoint(x=lwx, y=lwy, confidence=1.0),
            "right_wrist": Keypoint(x=rwx, y=rwy, confidence=1.0),
        }
        angles = PoseFeatureService().extract_joint_angles(
            KeypointCollection(keypoints=kps)
        )
        return score_form(angles, "shoulder_press")

    _SHOULDERS = [(0.40, 0.40), (0.60, 0.40)]
    _HIPS = [(0.42, 0.75), (0.58, 0.75)]
    # Lockout: wrists straight above the shoulders.
    _OVERHEAD = [(0.40, 0.10), (0.60, 0.10)]
    # Lateral/front raise at shoulder height — arms out, not pressed up.
    _ARM_RAISE = [(0.15, 0.40), (0.85, 0.40)]

    def test_overhead_press_scores_full(self):
        accuracy, suggestions = self._score(
            self._SHOULDERS, self._HIPS, self._OVERHEAD
        )
        assert accuracy == pytest.approx(1.0)
        assert suggestions == []

    def test_arm_raise_scores_lower_with_a_cue(self):
        good, _ = self._score(self._SHOULDERS, self._HIPS, self._OVERHEAD)
        bad, cues = self._score(self._SHOULDERS, self._HIPS, self._ARM_RAISE)
        assert bad < good
        assert cues
