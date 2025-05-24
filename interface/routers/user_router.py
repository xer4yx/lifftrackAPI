from fastapi import APIRouter, Path, Depends, status, HTTPException

from core.entities.user_entity import UserEntity
from core.usecase import UserUseCase
from interface.di import get_user_service_rest, get_current_user
from lifttrack.models.user_schema import User, UserUpdate, UserResponse, UserCreateResponse


# Create user router
user_router = APIRouter(
    prefix="/v3/user",
    tags=["v3-user"],
    responses={
        status.HTTP_401_UNAUTHORIZED: {"description": "Authentication failed"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_403_FORBIDDEN: {"description": "Access forbidden"}
    }
)


@user_router.post("", status_code=status.HTTP_201_CREATED, response_model=UserCreateResponse)
async def register_user(
    user_data: User,
    user_service: UserUseCase = Depends(get_user_service_rest)
):
    """
    Register a new user in the system.
    
    This endpoint is not secured to allow new user registration.
    """
    success, user, error = await user_service.register_user(
        username=user_data.username,
        email=user_data.email,
        password=user_data.password,
        first_name=user_data.fname,
        last_name=user_data.lname,
        phone_number=user_data.phoneNum,
        profile_picture=user_data.pfp
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=error or "Failed to register user"
        )
        
    # Create a UserCreateResponse object from the UserEntity
    return UserCreateResponse.model_validate(
        obj=user.model_dump(exclude={"created_at", "updated_at", "last_login", "password"}), 
        strict=False
    )


@user_router.get("/profile", response_model=UserResponse)
async def get_profile(
    current_user: UserEntity = Depends(get_current_user)
):
    """
    Get the current user's profile.
    
    This endpoint is secured and returns the authenticated user's information.
    """
    # Create a User object from the UserEntity, explicitly excluding password
    return UserResponse.model_validate(
        obj=current_user.model_dump(exclude={"password"}), 
        strict=False
    )


@user_router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: str,
    current_user: UserEntity = Depends(get_current_user),
    user_service: UserUseCase = Depends(get_user_service_rest)
):
    """
    Get a user by ID.
    
    This endpoint is secured and only allows access if the requested user ID
    matches the authenticated user ID.
    """
    # Security check: Only allow users to access their own data
    if current_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this user's data"
        )
    
    user = await user_service.get_user_profile(current_user.username)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
        
    # Create a UserResponse object from the UserEntity, explicitly excluding password
    return UserResponse.model_validate(
        obj=user.model_dump(exclude={"password"}), 
        strict=False
    )


@user_router.put("/{user_id}", response_model=UserCreateResponse)
async def update_user(
    user_id: str,
    update_data: UserUpdate,
    current_user: UserEntity = Depends(get_current_user),
    user_service: UserUseCase = Depends(get_user_service_rest)
):
    """
    Update a user's information.
    
    This endpoint is secured and only allows users to update their own data.
    """
    # Security check: Only allow users to update their own data
    if current_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this user's data"
        )
    
    # Convert UserUpdate to dict and map field names to match UserEntity
    user_update_dict = {}
    
    if update_data.fname is not None:
        user_update_dict["first_name"] = update_data.fname
    if update_data.lname is not None:
        user_update_dict["last_name"] = update_data.lname
    if update_data.email is not None:
        user_update_dict["email"] = update_data.email
    if update_data.phoneNum is not None:
        user_update_dict["phone_number"] = update_data.phoneNum
    if update_data.pfp is not None:
        user_update_dict["profile_picture"] = update_data.pfp
    
    # Password updates should be handled by the change password endpoint
    if update_data.password is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot update password through this endpoint. Use the change password endpoint."
        )
    
    success, updated_user, error = await user_service.update_user_profile(
        user_id=current_user.username,
        update_data=user_update_dict
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error or "Failed to update user"
        )
        
    # Create a UserCreateResponse object which doesn't require password
    return UserCreateResponse.model_validate(
        obj=updated_user.model_dump(exclude={"password", "created_at", "updated_at", "last_login"}), 
        strict=False
    )
    


@user_router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: str,
    current_user: UserEntity = Depends(get_current_user),
    user_service: UserUseCase = Depends(get_user_service_rest)
):
    """
    Delete a user.
    
    This endpoint is secured and only allows users to delete their own account.
    """
    # Security check: Only allow users to delete their own account
    if current_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this user"
        )
    
    success, error = await user_service.delete_user(current_user.username)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error or "Failed to delete user"
        )
    
    return None


@user_router.post("/{user_id}/change-password", status_code=status.HTTP_204_NO_CONTENT)
async def change_password(
    user_id: str,
    old_password: str,
    new_password: str,
    current_user: UserEntity = Depends(get_current_user),
    user_service: UserUseCase = Depends(get_user_service_rest)
):
    """
    Change a user's password.
    
    This endpoint is secured and only allows users to change their own password.
    """
    # Security check: Only allow users to change their own password
    if current_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to change this user's password"
        )
    
    success, error = await user_service.change_user_password(
        user_id=current_user.username,
        old_password=old_password,
        new_password=new_password
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error or "Failed to change password"
        )
    
    return None 