from lifttrack.models import User, Token, AppInfo, Frame
from lifttrack import timedelta
from lifttrack.auth import (create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES,
                            get_current_user)

from rtdbHelper import RTDBHelper
import threading

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm


app = FastAPI()
rtdb = RTDBHelper()

latest_frame_lock = threading.Lock()
latest_frame = None


@app.get("/")
def read_root():
    """
    Root endpoint.
    """
    return {"msg": "Welcome to LiftTrack!"}


@app.get("/app_info")
def get_app_info(app: AppInfo):
    """
    Endpoint to get information about the app.
    """
    return {
        "app_name": app.app_name,
        "version": app.version,
        "description": app.description
    }


@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = get_user_data(
        username=form_data.username,
        data=form_data.password
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": form_data.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


# API Endpoint [USER]
@app.get("/users/me/")
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user


@app.post("/{user}/track")
async def update_frame(frame: Frame):
    """
    Endpoint to receive frames from the mobile camera, perform pose estimation,
    and return the frame with keypoints drawn on it.
    """
    raise HTTPException(status_code=501, detail="Not Implemented")
    # global latest_frame
    #
    # user = frame.user
    # original_frame = frame.original_frame
    # still_image = frame.image
    #
    # try:
    #     ...
    #     image = cast_image(still_image)
    #     keypoints = run_movenet_inference(image)
    #
    # except HTTPException as httpe:
    #     return {
    #         "status": httpe.status_code,
    #         "detail": httpe.detail
    #     }


@app.put("/user/create")
def create_user(user: User):
    """
    Endpoint to create a new user in the Firebase Realtime Database.
    """
    try:
        user_data = {
            "id": user.id,
            "fname": user.fname,
            "lname": user.lname,
            "username": user.username,
            "phoneNum": user.phoneNum,
            "email": user.email,
            "password": user.password,
            "pfp": user.pfp,
            "isAuthenticated": user.isAuthenticated,
            "isDeleted": user.isDeleted
        }

        rtdb.put_data(user_data)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except TypeError as te:
        raise HTTPException(status_code=400, detail=str(te))


@app.get("/user/{username}")
async def get_user_data(username: str, data=None):
    """
    Endpoint to get user data from the Firebase Realtime Database.
    """
    try:
        user_data = rtdb.get_data(username, data)
        return user_data
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))


@app.put("/user/{username}")
async def update_user_data(username: str, user: User):
    """
    Endpoint to update user data in the Firebase Realtime Database.
    """
    try:
        user_data = {
            "id": user.id,
            "fname": user.fname,
            "lname": user.lname,
            "username": user.username,
            "phoneNum": user.phoneNum,
            "email": user.email,
            "password": user.password,
            "pfp": user.pfp,
            "isAuthenticated": user.isAuthenticated,
            "isDeleted": user.isDeleted
        }

        rtdb.update_data(username, user_data)
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))


@app.delete("/user/{username}")
def delete_user(username: str):
    """
    Endpoint to delete a user from the Firebase Realtime Database.
    """
    try:
        rtdb.delete_data(username)
        return {"msg": "User deleted"}
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
