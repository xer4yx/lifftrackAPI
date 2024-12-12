from typing import List, Optional

from lifttrack.models import User
from lifttrack.v2.dbhelper import get_db, FirebaseDBHelper

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from slowapi import Limiter
from slowapi.util import get_remote_address, get_ipaddr

router = APIRouter(
    prefix="/v2",
    tags=["v2-user"],
    responses={404: {"description": "Not found"}}
)


@router.post("/users", response_model=User, status_code=201)
async def create_user(user: User, db: FirebaseDBHelper = Depends(get_db)):
    """
        Create a new user in the Firebase database.
        
        Args:
            user (User): User creation details
            db (FirebaseDBHelper): Database connection
        
        Returns:
            User: Created user details
    """
    try:
        user_data = user.model_dump()
        existing_users = db.query_collection(
            'users',
            filters=[('username','==',user.username)]
        )
        
        if existing_users:
            raise HTTPException(status_code=400, detail="Username already exists")
        
        # Add user to Firebase
        user_id = db.add_document('users', user_data)
        
        return JSONResponse(
            content=User
        )
    except Exception as e:
        pass


@router.get("/users", response_model=List[User])
async def list_users(
    db: FirebaseDBHelper = Depends(get_db),
    is_authenticated: Optional[int] = None,
    is_deleted: Optional[int] = None
):
    """
    Retrieve users with optional age filtering.
    
    Args:
        db (FirebaseDBHelper): Database connection
        age_min (Optional[int]): Minimum age filter
        age_max (Optional[int]): Maximum age filter
    
    Returns:
        List[User]: List of user details
    """
    try:
        filters = []
        if is_authenticated is not None:
            filters.append(('isAuthenticated', '==', is_authenticated))
        if is_deleted is not None:
            filters.append(('isDeleted', '==', is_deleted))
        
        users_data = db.query_collection(
            'users', 
            filters=filters if filters else None,
            order_by='username'
        )
        
        return [
            User(
                user_id=user.get('id', ''),
                username=user['username'],
                email=user['email'],
            ).model_dump_json() for user in users_data
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve users: {str(e)}")


@router.put("/users/{user_id}")
async def update_user(
    user_id: str, 
    update_data: User, 
    db: FirebaseDBHelper = Depends(get_db)
):
    """
    Update an existing user's information.
    
    Args:
        user_id (str): ID of user to update
        update_data (User): New user information
        db (FirebaseDBHelper): Database connection
    
    Returns:
        dict: Update confirmation
    """
    try:
        # Convert Pydantic model to dict, removing None values
        update_dict = {k: v for k, v in update_data.model_dump().items() if v is not None}
        
        # Perform update
        update_result = db.update_document('users', user_id, update_dict)
        
        if not update_result:
            raise HTTPException(status_code=404, detail="User not found")
        
        return JSONResponse({"message": "User updated successfully"})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"User update failed: {str(e)}")


@router.delete("/users/{user_id}")
async def delete_user(user_id: str, db: FirebaseDBHelper = Depends(get_db)):
    """
    Delete a user from the database.
    
    Args:
        user_id (str): ID of user to delete
        db (FirebaseDBHelper): Database connection
    
    Returns:
        dict: Deletion confirmation
    """
    try:
        delete_result = db.delete_document('users', user_id)
        
        if not delete_result:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {"message": "User deleted successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"User deletion failed: {str(e)}")
