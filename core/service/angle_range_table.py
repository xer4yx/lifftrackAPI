"""Per-exercise angle-range table: the single source of truth for heatmap
bands, form score, and coaching suggestions (ADR 0005 + 0004).

Each form-relevant joint angle has a correct range ``[lo, hi]`` and an orange
``tolerance``. A measured angle inside the range is GREEN; within the tolerance
past either edge is ORANGE; beyond that is RED. The band is surfaced at the
``vertex`` keypoint (e.g. an elbow angle is colored at the elbow).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


class Band(str, Enum):
    GREEN = "green"
    ORANGE = "orange"
    RED = "red"


@dataclass(frozen=True)
class AngleRange:
    joint_angle: str  # key as produced by PoseFeatureService.extract_joint_angles
    vertex: str  # keypoint where the band is colored on the heatmap
    lo: float
    hi: float
    tolerance: float  # orange tolerance past either edge of [lo, hi]
    cue: str  # coaching suggestion shown when the angle is out of range


def classify_band(value: float, angle_range: AngleRange) -> Band:
    """Band a measured angle against its correct range and tolerance."""
    if angle_range.lo <= value <= angle_range.hi:
        return Band.GREEN
    if (
        angle_range.lo - angle_range.tolerance
        <= value
        <= angle_range.hi + angle_range.tolerance
    ):
        return Band.ORANGE
    return Band.RED


def _elbow(side: str) -> AngleRange:
    return AngleRange(
        joint_angle=f"{side}_shoulder_{side}_elbow_{side}_wrist",
        vertex=f"{side}_elbow",
        lo=75.0,
        hi=110.0,
        tolerance=15.0,
        cue="Keep your elbows around 90 degrees through the press.",
    )


def _overhead_press(side: str) -> AngleRange:
    # Score the arm *elevation* (wrist-shoulder-hip), not the elbow flexion: a
    # real press drives the arm overhead (~150-180 deg at lockout), while an
    # arbitrary partial arm-raise stays near 90 deg. The old elbow band
    # (30-100 deg) greened almost any bent arm, so form never discriminated
    # (ADR 0013 anti-degeneracy).
    return AngleRange(
        joint_angle=f"{side}_wrist_{side}_shoulder_{side}_hip",
        vertex=f"{side}_shoulder",
        lo=150.0,
        hi=180.0,
        tolerance=30.0,
        cue="Press the weight fully overhead — arms straight above your shoulders.",
    )


def _flat_back(side: str) -> AngleRange:
    return AngleRange(
        joint_angle=f"{side}_shoulder_{side}_hip_{side}_knee",
        vertex=f"{side}_hip",
        lo=150.0,
        hi=180.0,
        tolerance=15.0,
        cue="Keep your back flat; hips and shoulders rise together.",
    )


def _hip_hinge(side: str) -> AngleRange:
    return AngleRange(
        joint_angle=f"{side}_shoulder_{side}_hip_{side}_knee",
        vertex=f"{side}_hip",
        lo=60.0,
        hi=140.0,
        tolerance=15.0,
        cue="Push your hips back and keep your back flat.",
    )


def _neutral_head(side: str) -> AngleRange:
    return AngleRange(
        joint_angle=f"{side}_ear_{side}_shoulder_{side}_hip",
        vertex=f"{side}_shoulder",
        lo=160.0,
        hi=180.0,
        tolerance=15.0,
        cue="Keep your head neutral and eyes forward.",
    )


# Canonical per-exercise angle ranges. The angle keys match those emitted by
# PoseFeatureService.extract_joint_angles; the vertex is where the band is
# colored on the heatmap.
EXERCISE_ANGLE_RANGES: Dict[str, List[AngleRange]] = {
    "bench_press": [_elbow("left"), _elbow("right")],
    "deadlift": [
        _flat_back("left"),
        _flat_back("right"),
        _neutral_head("left"),
        _neutral_head("right"),
    ],
    "rdl": [
        _hip_hinge("left"),
        _hip_hinge("right"),
        _neutral_head("left"),
        _neutral_head("right"),
    ],
    "shoulder_press": [_overhead_press("left"), _overhead_press("right")],
}

# Name aliases so every exercise variant the pipeline recognizes resolves to the
# canonical ranges instead of returning nothing.
_ALIASES = {
    "benchpress": "bench_press",
    "romanian_deadlift": "rdl",
    "overhead_press": "shoulder_press",
}


def get_angle_ranges(exercise_name: str) -> List[AngleRange]:
    """Return the angle ranges for an exercise (empty list if unknown)."""
    key = exercise_name.lower().replace(" ", "_")
    key = _ALIASES.get(key, key)
    return EXERCISE_ANGLE_RANGES.get(key, [])


@dataclass(frozen=True)
class AngleAssessment:
    joint_angle: str
    vertex: str
    value: float
    band: Band
    cue: Optional[str]  # None when the band is green


# Per-band form-score deduction. Green is on-target; orange is a slight
# deviation; red is a severe deviation (ADR 0004).
_BAND_DEDUCTION = {Band.GREEN: 0.0, Band.ORANGE: 0.1, Band.RED: 0.2}


def assess_angles(
    joint_angles: Dict[str, float], exercise_name: str
) -> List[AngleAssessment]:
    """Band each of the exercise's form-relevant angles that is present.

    Missing angles are skipped rather than defaulted, so an occluded keypoint
    never produces a spurious band.
    """
    assessments: List[AngleAssessment] = []
    for r in get_angle_ranges(exercise_name):
        if r.joint_angle not in joint_angles:
            continue
        value = joint_angles[r.joint_angle]
        band = classify_band(value, r)
        assessments.append(
            AngleAssessment(
                joint_angle=r.joint_angle,
                vertex=r.vertex,
                value=value,
                band=band,
                cue=None if band is Band.GREEN else r.cue,
            )
        )
    return assessments


def score_form(
    joint_angles: Dict[str, float], exercise_name: str
) -> Tuple[float, List[str]]:
    """Derive a form accuracy (0-1) and de-duplicated suggestions from the table."""
    assessments = assess_angles(joint_angles, exercise_name)
    deduction = sum(_BAND_DEDUCTION[a.band] for a in assessments)
    accuracy = max(0.0, min(1.0, 1.0 - deduction))

    suggestions: List[str] = []
    for a in assessments:
        if a.cue and a.cue not in suggestions:
            suggestions.append(a.cue)
    return accuracy, suggestions
