from admin_rtdb import FirebaseDBHelper
from fastapi import HTTPException

def get_db():
    """
    Dependency injection for database connection.
    Ensures a single database instance is used across requests.
    """
    try:
        # Initialize Firebase with credentials path
        db = FirebaseDBHelper('path/to/firebase_credentials.json')
        return db
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")
