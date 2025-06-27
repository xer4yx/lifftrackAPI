import numpy as np
import math
from typing import Dict, Any, List, Optional, Tuple
import concurrent.futures

from core.interface.pose_feature_interface import PoseFeatureInterface
from core.entities.pose_entity import (
    Keypoint,
    KeypointCollection,
    BodyAlignment,
    PoseFeatures,
)

from lifttrack.utils.logging_config import setup_logger

logger = setup_logger("service.posefeature", "core.log")


class PoseFeatureService(PoseFeatureInterface):
    def __init__(self):
        self.max_workers = 4  # For concurrent processing

    def calculate_angle(
        self, a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]
    ) -> float:
        """Calculates the angle formed by three points a, b, c."""
        # Convert points to numpy arrays for easier calculation
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        # Vectors BA and BC
        ba = a - b
        bc = c - b

        # Compute the cosine of the angle using dot product
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))

        # Avoid domain errors due to floating point precision
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        # Return the angle in degrees
        angle = np.arccos(cos_angle) * (180.0 / np.pi)
        return angle

    def extract_joint_angles(
        self, keypoints: KeypointCollection, confidence_threshold: float = 0.1
    ) -> Dict[str, float]:
        """
        Extracts joint angles for all keypoints, filtered by confidence.
        """
        # Convert KeypointCollection to the format expected by the original function
        kp_dict = {
            k: (kp.x, kp.y, kp.confidence) for k, kp in keypoints.keypoints.items()
        }

        # Filter keypoints by confidence
        filtered_keypoints = {
            k: v for k, v in kp_dict.items() if v[2] >= confidence_threshold
        }

        # Add more exercise-specific joint pairs for better form analysis
        joint_pairs = [
            ("left_shoulder", "left_elbow", "left_wrist"),
            ("right_shoulder", "right_elbow", "right_wrist"),
            ("left_hip", "left_knee", "left_ankle"),
            ("right_hip", "right_knee", "right_ankle"),
            ("left_shoulder", "left_hip", "left_knee"),
            ("right_shoulder", "right_hip", "right_knee"),
            ("left_shoulder", "right_shoulder", "neck"),
            ("left_hip", "right_hip", "waist"),
            # Add these new pairs for better exercise form analysis
            ("left_ear", "left_shoulder", "left_hip"),  # Head alignment for deadlift
            ("right_ear", "right_shoulder", "right_hip"),  # Head alignment for deadlift
            (
                "left_shoulder",
                "left_hip",
                "left_ankle",
            ),  # Back-to-ankle alignment for deadlift
            (
                "right_shoulder",
                "right_hip",
                "right_ankle",
            ),  # Back-to-ankle alignment for deadlift
            ("left_wrist", "left_shoulder", "left_hip"),  # Bar path for bench press
            ("right_wrist", "right_shoulder", "right_hip"),  # Bar path for bench press
        ]

        angles = {}

        for pair in joint_pairs:
            joint1, joint2, joint3 = pair
            if (
                joint1 in filtered_keypoints
                and joint2 in filtered_keypoints
                and joint3 in filtered_keypoints
            ):
                angle = self.calculate_angle(
                    filtered_keypoints[joint1],
                    filtered_keypoints[joint2],
                    filtered_keypoints[joint3],
                )
                angles[f"{joint1}_{joint2}_{joint3}"] = angle

        return angles

    def extract_movement_patterns(
        self, keypoints: KeypointCollection, previous_keypoints: KeypointCollection
    ) -> Dict[str, float]:
        """
        Tracks the movement pattern of joints by calculating displacement from previous keypoints.
        """
        displacement = {}

        for joint_name, keypoint in keypoints.keypoints.items():
            prev_keypoint = previous_keypoints.get(joint_name)
            if prev_keypoint:
                curr_pos = np.array((keypoint.x, keypoint.y))
                prev_pos = np.array((prev_keypoint.x, prev_keypoint.y))
                dist = np.linalg.norm(curr_pos - prev_pos)
                displacement[joint_name] = dist

        logger.info(f"Displacement: {displacement}")

        return displacement

    def calculate_speed(
        self, displacement: Dict[str, float], time_delta: float = 1.0
    ) -> Dict[str, float]:
        """
        Computes the speed of movement based on displacement and time delta.
        """
        speed = {}
        for joint in displacement:
            speed[joint] = displacement[joint] / time_delta
        return speed

    def extract_body_alignment(
        self, keypoints: KeypointCollection
    ) -> Tuple[float, float]:
        """
        Calculate body alignment by measuring the angle between shoulders and hips.
        """
        alignment = [0, 0]  # Default values

        # Get keypoints for required joints
        left_shoulder = keypoints.get("left_shoulder")
        right_shoulder = keypoints.get("right_shoulder")
        left_hip = keypoints.get("left_hip")
        right_hip = keypoints.get("right_hip")

        if left_shoulder and right_shoulder and left_hip and right_hip:
            # Convert to numpy arrays
            ls_pos = np.array((left_shoulder.x, left_shoulder.y))
            rs_pos = np.array((right_shoulder.x, right_shoulder.y))
            lh_pos = np.array((left_hip.x, left_hip.y))
            rh_pos = np.array((right_hip.x, right_hip.y))

            # Calculate midpoints
            shoulder_midpoint = (ls_pos + rs_pos) / 2
            hip_midpoint = (lh_pos + rh_pos) / 2

            # Calculate vertical alignment (angle with vertical axis)
            vertical_vector = np.array([0, 1])  # Vertical reference vector
            body_vector = shoulder_midpoint - hip_midpoint

            # Normalize vectors
            vertical_vector = vertical_vector / np.linalg.norm(vertical_vector)
            if np.linalg.norm(body_vector) > 0:
                body_vector = body_vector / np.linalg.norm(body_vector)

                # Calculate angle between body vector and vertical
                cos_angle = np.dot(body_vector, vertical_vector)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                vertical_alignment = np.arccos(cos_angle) * (180.0 / np.pi)

                # Calculate lateral tilt (left-right balance)
                shoulder_vector = rs_pos - ls_pos
                hip_vector = rh_pos - lh_pos

                if (
                    np.linalg.norm(shoulder_vector) > 0
                    and np.linalg.norm(hip_vector) > 0
                ):
                    shoulder_vector = shoulder_vector / np.linalg.norm(shoulder_vector)
                    hip_vector = hip_vector / np.linalg.norm(hip_vector)

                    cos_lateral = np.dot(shoulder_vector, hip_vector)
                    cos_lateral = np.clip(cos_lateral, -1.0, 1.0)
                    lateral_alignment = np.arccos(cos_lateral) * (180.0 / np.pi)

                    alignment = [vertical_alignment, lateral_alignment]

        return tuple(alignment)

    def calculate_stability(
        self,
        keypoints: KeypointCollection,
        previous_keypoints: KeypointCollection,
        window_size: int = 5,
    ) -> float:
        """
        Calculate stability as the total displacement of core keypoints over time.
        Lower values indicate better stability.
        """
        # Core keypoints that are most relevant for stability
        core_joints = [
            "left_shoulder",
            "right_shoulder",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
        ]

        # If we don't have previous keypoints, just return 0
        if not previous_keypoints or not previous_keypoints.keypoints:
            return 0.0

        total_displacement = 0.0
        count = 0

        # Focus on core joints for stability calculation
        for joint_name in core_joints:
            keypoint = keypoints.get(joint_name)
            prev_keypoint = previous_keypoints.get(joint_name)

            if keypoint and prev_keypoint:
                curr_pos = np.array((keypoint.x, keypoint.y))
                prev_pos = np.array((prev_keypoint.x, prev_keypoint.y))

                # Calculate displacement
                displacement = np.linalg.norm(curr_pos - prev_pos)

                # Weight displacement by joint importance
                if "shoulder" in joint_name or "hip" in joint_name:
                    # Core joints have higher weight
                    total_displacement += displacement * 1.5
                else:
                    total_displacement += displacement

                count += 1

        # Normalize by number of joints to get average displacement
        if count > 0:
            return total_displacement / count
        return 0.0

    def detect_form_issues(
        self, features: PoseFeatures, exercise_type: str
    ) -> Dict[str, bool]:
        """
        Detect common form issues for specific exercises.
        """
        issues = {}
        angles = features.joint_angles

        if exercise_type == "bench_press" or exercise_type == "benchpress":
            # Check for wrist alignment
            left_wrist_angle = abs(
                90 - angles.get("left_shoulder_left_elbow_left_wrist", 90)
            )
            right_wrist_angle = abs(
                90 - angles.get("right_shoulder_right_elbow_right_wrist", 90)
            )

            if left_wrist_angle > 20 or right_wrist_angle > 20:
                issues["wrist_alignment"] = True

            # Check for elbow position
            left_elbow = angles.get("left_shoulder_left_elbow_left_wrist", 180)
            right_elbow = angles.get("right_shoulder_right_elbow_right_wrist", 180)

            if left_elbow > 110 or right_elbow > 110:
                issues["elbow_position"] = True

        elif (
            exercise_type == "deadlift"
            or exercise_type == "romanian_deadlift"
            or exercise_type == "rdl"
        ):
            # Check for back angle
            back_angle = angles.get("left_shoulder_left_hip_left_knee", 180)

            if exercise_type == "deadlift" and back_angle and back_angle < 150:
                issues["back_angle"] = True
            elif (
                (exercise_type == "romanian_deadlift" or exercise_type == "rdl")
                and back_angle
                and (back_angle > 140 or back_angle < 60)
            ):
                issues["hip_hinge"] = True

            # Check for head position
            head_angle = angles.get("left_ear_left_shoulder_left_hip", 180)
            if head_angle and abs(head_angle - 180) > 20:
                issues["head_position"] = True

        return issues

    def detect_resting_state(
        self, keypoints: KeypointCollection, features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Detect if the user is in a resting state (standing or sitting).
        """
        result = {"is_resting": False, "position": "unknown", "confidence": 0.0}

        # Check if there are any objects detected
        objects = features.get("objects", {})
        if not objects:
            result["is_resting"] = True
            result["confidence"] = 0.9
            return result

        # Get stability measure
        stability = features.get("stability", 0)

        # Calculate angles concurrently
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            # Extract keypoint positions for calculations
            left_hip = keypoints.get("left_hip")
            left_knee = keypoints.get("left_knee")
            left_ankle = keypoints.get("left_ankle")
            right_hip = keypoints.get("right_hip")
            right_knee = keypoints.get("right_knee")
            right_ankle = keypoints.get("right_ankle")
            left_shoulder = keypoints.get("left_shoulder")
            right_shoulder = keypoints.get("right_shoulder")

            # Define angle calculation tasks
            angle_tasks = {
                "left_leg": (
                    executor.submit(
                        self.calculate_angle,
                        (left_hip.x, left_hip.y) if left_hip else (0, 0),
                        (left_knee.x, left_knee.y) if left_knee else (0, 0),
                        (left_ankle.x, left_ankle.y) if left_ankle else (0, 0),
                    )
                    if all([left_hip, left_knee, left_ankle])
                    else None
                ),
                "right_leg": (
                    executor.submit(
                        self.calculate_angle,
                        (right_hip.x, right_hip.y) if right_hip else (0, 0),
                        (right_knee.x, right_knee.y) if right_knee else (0, 0),
                        (right_ankle.x, right_ankle.y) if right_ankle else (0, 0),
                    )
                    if all([right_hip, right_knee, right_ankle])
                    else None
                ),
                "left_back": (
                    executor.submit(
                        self.calculate_angle,
                        (left_shoulder.x, left_shoulder.y) if left_shoulder else (0, 0),
                        (left_hip.x, left_hip.y) if left_hip else (0, 0),
                        (left_knee.x, left_knee.y) if left_knee else (0, 0),
                    )
                    if all([left_shoulder, left_hip, left_knee])
                    else None
                ),
                "right_back": (
                    executor.submit(
                        self.calculate_angle,
                        (
                            (right_shoulder.x, right_shoulder.y)
                            if right_shoulder
                            else (0, 0)
                        ),
                        (right_hip.x, right_hip.y) if right_hip else (0, 0),
                        (right_knee.x, right_knee.y) if right_knee else (0, 0),
                    )
                    if all([right_shoulder, right_hip, right_knee])
                    else None
                ),
            }

            # Get results with error handling
            try:
                left_hip_knee_ankle = (
                    angle_tasks["left_leg"].result()
                    if angle_tasks["left_leg"]
                    else None
                )
                right_hip_knee_ankle = (
                    angle_tasks["right_leg"].result()
                    if angle_tasks["right_leg"]
                    else None
                )
                left_back_angle = (
                    angle_tasks["left_back"].result()
                    if angle_tasks["left_back"]
                    else None
                )
                right_back_angle = (
                    angle_tasks["right_back"].result()
                    if angle_tasks["right_back"]
                    else None
                )
            except Exception:
                left_hip_knee_ankle = right_hip_knee_ankle = left_back_angle = (
                    right_back_angle
                ) = None

        # Use the more reliable back angle (average if both available)
        back_angle = None
        if left_back_angle is not None and right_back_angle is not None:
            back_angle = (left_back_angle + right_back_angle) / 2
        elif left_back_angle is not None:
            back_angle = left_back_angle
        elif right_back_angle is not None:
            back_angle = right_back_angle

        # Check for sitting positions
        # Floor sitting: knees more bent (90-120 degrees), back more upright (70-90 degrees)
        is_sitting_floor = (
            (left_hip_knee_ankle and 90 <= left_hip_knee_ankle <= 120)
            or (right_hip_knee_ankle and 90 <= right_hip_knee_ankle <= 120)
        ) and (back_angle and 70 <= back_angle <= 90)

        # Chair sitting: knees less bent (70-100 degrees), back more reclined (90-120 degrees)
        is_sitting_chair = (
            (left_hip_knee_ankle and 70 <= left_hip_knee_ankle <= 100)
            or (right_hip_knee_ankle and 70 <= right_hip_knee_ankle <= 100)
        ) and (back_angle and 90 <= back_angle <= 120)

        # Check for standing position
        # Standing: straighter legs (160-180 degrees), more upright back (150-180 degrees)
        is_standing = (
            (left_hip_knee_ankle and 160 <= left_hip_knee_ankle <= 180)
            or (right_hip_knee_ankle and 160 <= right_hip_knee_ankle <= 180)
        ) and (back_angle and 150 <= back_angle <= 180)

        # High stability indicates less movement
        is_stable = stability < 5.0  # Low displacement indicates stillness

        if is_stable and not objects:
            result["is_resting"] = True
            result["confidence"] = 0.8

            # Calculate confidence based on how well angles match expected ranges
            if is_sitting_floor and not objects:
                result["position"] = "sitting_floor"
                # Higher confidence if both legs match the pattern
                confidence = (
                    0.9 if (left_hip_knee_ankle and right_hip_knee_ankle) else 0.8
                )
                result["confidence"] = confidence

            elif is_sitting_chair and not objects:
                result["position"] = "sitting_chair"
                # Higher confidence if both legs match the pattern
                confidence = (
                    0.9 if (left_hip_knee_ankle and right_hip_knee_ankle) else 0.8
                )
                result["confidence"] = confidence

            elif is_standing and not objects:
                result["position"] = "standing"
                # Higher confidence if both legs are straight
                confidence = (
                    0.9 if (left_hip_knee_ankle and right_hip_knee_ankle) else 0.8
                )
                result["confidence"] = confidence

        return result

    def process_features(
        self,
        keypoints: KeypointCollection,
        previous_keypoints: Optional[KeypointCollection] = None,
        objects: Dict[str, Any] = None,
    ) -> PoseFeatures:
        """
        Process keypoints to extract all features in one call.
        """
        # Extract joint angles
        joint_angles = self.extract_joint_angles(keypoints)

        # Initialize with default values
        movement_patterns = {}
        speed = {}
        stability = 0.0

        # Extract movement patterns and speed if we have previous keypoints
        if previous_keypoints:
            movement_patterns = self.extract_movement_patterns(
                keypoints, previous_keypoints
            )
            speed = self.calculate_speed(movement_patterns)
            stability = self.calculate_stability(keypoints, previous_keypoints)

        # Extract body alignment
        vertical_alignment, lateral_alignment = self.extract_body_alignment(keypoints)
        body_alignment = BodyAlignment(vertical_alignment, lateral_alignment)

        # Create PoseFeatures object
        features = PoseFeatures(
            keypoints=keypoints,
            joint_angles=joint_angles,
            movement_patterns=movement_patterns,
            body_alignment=body_alignment,
            stability=stability,
            speed=speed,
            objects=objects or {},
        )

        return features
