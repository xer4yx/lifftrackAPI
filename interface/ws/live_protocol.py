"""Builders for the live wire-contract messages (contract/schemas).

The scoring itself lives in the angle-range table (ADR 0005); these helpers only
shape the results into the ``bands`` and ``coaching`` messages defined by the
wire contract (ADR 0011).
"""

from typing import Dict, List

from core.service.angle_range_table import assess_angles, score_form


def build_bands_message(
    joint_angles: Dict[str, float], exercise_name: str, t: int
) -> dict:
    """Build a ``bands`` message: vertex keypoint -> color, echoing timestamp t.

    Colors only, no positions; the client paints them onto its own keypoints.
    """
    assessments = assess_angles(joint_angles, exercise_name)
    return {
        "type": "bands",
        "t": t,
        "bands": {a.vertex: a.band.value for a in assessments},
    }


def build_coaching_message(
    joint_angles: Dict[str, float], exercise_name: str
) -> dict:
    """Build a ``coaching`` message: form accuracy + de-duplicated suggestions."""
    accuracy, suggestions = score_form(joint_angles, exercise_name)
    return {
        "type": "coaching",
        "accuracy": accuracy,
        "suggestions": suggestions,
    }
