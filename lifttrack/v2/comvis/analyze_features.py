import os
import cv2
from lifttrack.v2.comvis.object_track import process_frames_and_get_annotations
from lifttrack.v2.comvis.features import (
    extract_joint_angles,
    extract_movement_patterns,
    calculate_speed,
    extract_body_alignment,
    calculate_stability
)


def analyze_annotations(frames_directory, analyze_frame_function):
    """
    Processes frames to extract annotations and computes features based on those annotations.
    """
    # Step 1: Process frames and get annotations
    annotations, final_annotated_frame = process_frames_and_get_annotations(frames_directory, analyze_frame_function)
    
    if not annotations:
        return [], None
    
    previous_keypoints = None
    all_features = []
    
    for annotation in annotations:
        try:
            keypoints = annotation['keypoints']
            objects = annotation.get('objects', [])  # Use get() with default empty list
            frame_path = annotation['frame_path']
            
            # Validate keypoints
            if not keypoints:
                print(f"Warning: No keypoints detected in frame {frame_path}")
                continue
            
            # Extract features with validation
            joint_angles = extract_joint_angles(keypoints)
            body_alignment = extract_body_alignment(keypoints)
            
            # Initialize movement-based features
            movement_features = {
                'movement_patterns': {},
                'speeds': {},
                'stability': 0.0
            }
            
            # Calculate movement-based features if we have previous keypoints
            if previous_keypoints is not None:
                movement_patterns = extract_movement_patterns(keypoints, previous_keypoints)
                movement_features.update({
                    'movement_patterns': movement_patterns,
                    'speeds': calculate_speed(movement_patterns),
                    'stability': calculate_stability(keypoints, previous_keypoints)
                })
            
            # Compile features with validation
            features = {
                'frame_path': frame_path,
                'objects': objects,
                'joint_angles': joint_angles or {},
                'movement_patterns': movement_features['movement_patterns'],
                'speeds': movement_features['speeds'],
                'body_alignment': body_alignment or {},
                'stability': movement_features['stability'],
                'timestamp': annotation.get('image_info', {}).get('timestamp')
            }
            
            all_features.append(features)
            previous_keypoints = keypoints
            
        except Exception as e:
            print(f"Error processing frame {annotation.get('frame_path', 'unknown')}: {str(e)}")
            continue
    
    return all_features, final_annotated_frame
