from .Movenet import MovenetInference
from .object_track import ObjectTracker
from .Live import ThreeDimInference

movenet_inference = MovenetInference()
object_tracker = ObjectTracker()
three_dim_inference = ThreeDimInference()

__all__ = ["MovenetInference", "ObjectTracker", "ThreeDimInference"]
