# Heatmap is banded form deviation, not injury prediction

**Status:** accepted

## Context

The heatmap colors body regions green / orange / red. The initial framing was
"normal / slight deviation / risk of injury," which reads as a clinical
injury-risk prediction. We have no labeled injury data, and 2D single-pose
keypoints cannot reliably detect real injury mechanisms (spinal flexion under
load, knee valgus) at arbitrary camera angles. A red band labeled "injury
risk" driven by an angle threshold would be an unsubstantiated medical claim.

## Decision

The heatmap is a **visualization of per-joint form deviation**, banded into
three colors by how far each measured joint angle sits from its
exercise-specific correct range. It reuses the existing form-analysis signals
rather than introducing a separate injury model.

The **domain model and API do not use the word "injury."** A red band means
"severe form deviation," which is honest, actionable coaching that correlates
with risk without asserting a clinical prediction. UI marketing copy wording is
a separate, deferred product decision.

## Consequences

- No new ML model or injury dataset is needed; the heatmap renders signals the
  scoring already produces (once the known angle bugs are fixed).
- Per-joint ideal ranges and band tolerances become the core data the heatmap
  depends on.
- If an injury-prediction claim is ever required, it needs its own model,
  validation, and ADR — it is explicitly out of scope here.
