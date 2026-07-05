import importlib.util
import pathlib

# Load the standalone data module directly by path. Importing it normally would
# run interface/ws/__init__.py, which eagerly pulls in the whole websocket router
# (FastAPI et al.) just to read a constant dict.
_constant_path = (
    pathlib.Path(__file__).resolve().parents[2] / "interface" / "ws" / "constant.py"
)
_spec = importlib.util.spec_from_file_location("_ws_constant", _constant_path)
_constant = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_constant)
EXERCISE_THRESHOLDS = _constant.EXERCISE_THRESHOLDS

# Exercise-name variants that the form-analysis pipeline
# (core.service.form_analysis_service.FormAnalysisService.analyze_form)
# recognizes and dispatches on. The threshold table must cover all of them,
# otherwise a recognized exercise silently scores against the default
# thresholds. (Step 3 / ADR 0005 folds this into one source-of-truth table.)
RECOGNIZED_EXERCISES = [
    "bench_press",
    "benchpress",
    "deadlift",
    "romanian_deadlift",
    "rdl",
    "shoulder_press",
    "overhead_press",
]

REQUIRED_KEYS = {
    "max_allowed_deviation",
    "max_allowed_variance",
    "max_jerk",
    "max_displacement",
}


def test_every_recognized_exercise_has_a_threshold_entry():
    missing = [e for e in RECOGNIZED_EXERCISES if e not in EXERCISE_THRESHOLDS]
    assert missing == [], (
        f"exercises with no threshold entry (silently fall back to default): {missing}"
    )


def test_every_threshold_entry_is_well_formed():
    for name, entry in EXERCISE_THRESHOLDS.items():
        assert REQUIRED_KEYS <= set(entry), f"{name} is missing keys {REQUIRED_KEYS - set(entry)}"
