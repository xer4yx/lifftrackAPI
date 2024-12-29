from typing import List, Optional

from lifttrack.models import User
from lifttrack.auth import get_password_hash
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
        user_data['password'] = get_password_hash(user_data['password'])
        
        # Check for existing username using RTDB query
        existing_data = db.query_data(
            'users',
            order_by='username',
            start_at=user.username,
            end_at=user.username
        )
        
        if existing_data:
            raise HTTPException(status_code=400, detail="Username already exists")
        
        # Add user to Firebase RTDB
        user_id = db.set_data(path='users', data=user_data, key=user_data['username'])
        user_data['id'] = user_id
        
        return JSONResponse(content=user_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create user: {str(e)}")


@router.get("/users", response_model=List[User])
async def list_users(
    db: FirebaseDBHelper = Depends(get_db),
    is_authenticated: Optional[bool] = False,
    is_deleted: Optional[bool] = False
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
        # Get all users first
        users_data = db.query_data(
            'users',
            order_by='username'
        )
        
        if not users_data:
            return []
        
        # Convert to list if it's a dictionary
        if isinstance(users_data, dict):
            users_list = [
                {'id': key, **value} 
                for key, value in users_data.items()
            ]
        else:
            users_list = users_data
            
        # Apply filters manually
        filtered_users = [
            user for user in users_list
            if user.get('isAuthenticated', False) == is_authenticated
            and user.get('isDeleted', False) == is_deleted
        ]
        
        return [
            User(
                user_id=user.get('id', ''),
                username=user.get('username', ''),
                email=user.get('email', ''),
                fname=user.get('fname', ''),  # Added required field
                lname=user.get('lname', ''),  # Added required field
                phoneNum=user.get('phoneNum', ''),  # Added required field
                password=user.get('password', ''),  # Added required field
                is_authenticated=user.get('isAuthenticated', False),
                is_deleted=user.get('isDeleted', False)
            ) for user in filtered_users
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
        # Check if user exists
        existing_user = db.get_data('users', user_id)
        if not existing_user:
            raise HTTPException(status_code=404, detail="User not found")
            
        # Convert Pydantic model to dict, removing None values
        update_dict = {k: v for k, v in update_data.model_dump().items() if v is not None}
        
        # Perform update
        success = db.update_data('users', user_id, update_dict)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update user")
        
        return JSONResponse({"message": "User updated successfully"})
    
    except HTTPException as he:
        raise he
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
        # Check if user exists
        existing_user = db.get_data('users', user_id)
        if not existing_user:
            raise HTTPException(status_code=404, detail="User not found")
            
        # Perform deletion
        success = db.delete_data('users', user_id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete user")
        
        return {"message": "User deleted successfully"}
    
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"User deletion failed: {str(e)}")
