# Per-exercise angle-range table as single source of truth

**Status:** accepted

## Context

The heatmap colors each form-relevant joint by how far its measured angle sits
from the correct range for the current exercise. Today those numbers are magic
constants scattered across `detect_form_issues` and the four `analyze_*_form`
methods, and duplicated again in the legacy `progress.py`. The heatmap needs
the same numbers, which would create a third copy.

## Decision

Each form-relevant angle is colored at its **vertex keypoint** (elbow angle at
the elbow, back angle at the hip, etc.). The band is computed from a
**per-exercise data table**: for each angle, a correct range `[lo, hi]` and an
orange tolerance. Inside the range → green; within tolerance past an edge →
orange; beyond → red.

This table is the **single source of truth** for the heatmap bands, the form
score, and the suggestions. The scattered constants and the duplicate rubric in
`progress.py` are replaced by reads from it.

## Consequences

- Adding or tuning an exercise is a data edit, not a code change across three
  files.
- Directly advances the maintainability goal: one table replaces the
  duplicated, drifting thresholds the code review flagged.
- The table's shape (angle names, ranges, tolerances) must cover every exercise
  the classifier/selector can produce, using the canonical exercise names.
