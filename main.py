import base64

import numpy as np

from lifttrack.rtdbHelper import *
from lifttrack.models import User, Token, AppInfo
from lifttrack.comvis import cv2, frame_queue, result_queue, generate_frames

from lifttrack import timedelta
from lifttrack.auth import (create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES,
                            get_current_user)

import threading

from fastapi import FastAPI, Depends, HTTPException, status, WebSocket
from fastapi import WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordRequestForm

server_origin = [
    'http://localhost:8000',
    'http://127.0.0.1:8000/'
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


latest_frame_lock = threading.Lock()
latest_frame = None

app = FastAPI()


# API Endpoint [ROOT]
@app.get("/")
def read_root():
    """
    Root endpoint.
    """
    return {"msg": "Welcome to LiftTrack!"}


# API Endpoint [About App]
@app.get("/app_info")
def get_app_info(appinfo: AppInfo):
    """
    Endpoint to get information about the app.
    """
    return appinfo


# API Endpoint [Authentication Operations]
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


# API Endpoint [RTDB Operations]
@app.get("/users/me/")
async def read_users_me(current_user: User = Depends(get_current_user)):
    """
    Endpoint to get the current user.
    """
    return current_user


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

        put_data(user_data)
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
        user_data = get_data(username, data)
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

        update_data(username, user_data)
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))


@app.delete("/user/{username}")
def delete_user(username: str):
    """
    Endpoint to delete a user from the Firebase Realtime Database.
    """
    try:
        delete_data(username)
        return {"msg": "User deleted"}
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))


# API Endpoint [Frame Operations]
# TODO: Implement roboflow inference and 3D CNN Inference for Web and Mobile versions
@app.websocket("/ws-tracking")  # Mobile version
async def websocket_inference(websocket: WebSocket):
    """
    Websocket endpoint for the MoveNet model.
    """
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_text()
            original_frame = base64.b64decode(data)
            frame = cv2.imdecode(np.frombuffer(original_frame, dtype=np.uint8), cv2.IMREAD_COLOR)

            if not frame_queue.full():
                frame_queue.put(frame)

            if not result_queue.empty():
                annotated_frame = result_queue.get()
                _, buffer = cv2.imencode(".jpg", annotated_frame)
                encoded_frame = base64.b64encode(buffer).decode('utf-8')
                await websocket.send_text(encoded_frame)
    except WebSocketDisconnect as wsde:
        return {
            "code": str(wsde.code),
            "msg": "WebSocket connection closed.",
            "details": str(wsde.reason)
        }, await websocket.close()


@app.get("/stream-tracking")
async def video_feed():  # Web version
    """
    Endpoint for the video feed.
    """
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
