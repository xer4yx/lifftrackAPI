from rtdbHelper import RTDBHelper

from datetime import datetime
from typing import Union
import threading

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


class User(BaseModel):
    id: str = datetime.strftime(datetime.now(), '%Y%H%d%m')
    fname: str
    lname: str
    username: str
    phoneNum: str
    email: str
    password: str
    pfp: str = None
    isAuthenticated: bool = False
    isDeleted: bool = False


class AppInfo(BaseModel):
    app_name: str = "LiftTrack"
    version: str = "1.0.0"
    description: str = "An app to track your lifts and provide feedback on your form."


class Frame(BaseModel):
    user: str
    original_frame: Union[int, int]
    image: bytes


class FormOutput(BaseModel):
    user: str
    current_reps: int
    num_errors: int


server_origin = [
    'http://localhost:8000'
]

server_method = ["PUT", "GET", "DELETE"]

server_header = ["*"]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=server_origin,
    allow_methods=server_method,
    allow_headers=server_header
)
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
def get_app_info():
    """
    Endpoint to get information about the app.
    """

    app = AppInfo()

    return {
        "app_name": app.app_name,
        "version": app.version,
        "description": app.description
    }


@app.post("/{user}/track")
async def update_frame(frame: Frame):
    """
    Endpoint to receive frames from the mobile camera, perform pose estimation,
    and return the frame with keypoints drawn on it.
    """
    raise HTTPException(status_code=501, detail="Not Implemented")
    global latest_frame

    user = frame.user
    original_frame = frame.original_frame
    still_image = frame.image

    try:
        ...
        # image = cast_image(still_image)
        # keypoints = run_movenet_inference(image)

    except HTTPException as httpe:
        return {
            "status": httpe.status_code,
            "detail": httpe.detail
        }


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

        return {"msg": "User created"}
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
