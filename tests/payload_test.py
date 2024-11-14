import requests

payload = {
    "squat": {
        "2023-10-01T12:00:00Z": {
            "suggestion": "Keep your back straight.",
            "features": {
                "objects": "person, barbell",
                "movement_pattern": 
                "squat","body_alignment": "aligned",
            "stability": "stable"
        },
        "frame": "frame_1"
      }
    }
}

requests.put("http://192.168.0.12:8000/progress/jdoeman")