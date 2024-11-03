from lifttrack import timedelta, threading, asyncio, base64
from lifttrack.dbhandler.rtdbHelper import rtdb
from lifttrack.models import User, Token, AppInfo, LoginForm
from lifttrack.comvis import cv2, websocket_process_frames
from lifttrack.auth import (create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES, get_current_user, verify_password,
                            get_password_hash)

from fastapi import FastAPI, Depends, HTTPException, status, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi import WebSocketDisconnect
from fastapi.security import OAuth2PasswordRequestForm

server_origin = [
    'http://localhost:8000',
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


# API Endpoint [ROOT]
@app.get("/")
async def read_root():
    """Lifttrack API root endpoint."""
    return {"msg": "Welcome to LiftTrack!"}


# API Endpoint [About App]
@app.get("/app-info")
async def get_app_info(appinfo: AppInfo):
    """Endpoint to get information about the app."""
    return appinfo


# API Endpoint [Authentication Operations]
@app.post("/login")
async def login(login_form: LoginForm):
    """
    API endpoint for user login.

    Args:
        login_form: BaseModel that contains username and password.
    """
    user_data = rtdb.get_data(login_form.username)

    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")

    if not verify_password(login_form.password, user_data["password"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    return {"message": "Login successful", "success": True}


@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Endpoint to get an access token.

    Args:
        form_data: OAuth2PasswordRequestForm that contains username and password.
    """
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

    Args:
        current_user: User model that contains user data.
    """
    return current_user


@app.put("/user/create")
def create_user(user: User):
    """
    Endpoint to create a new user in the Firebase Realtime Database.

    Args:
        user: User model that contains user data.
    """
    try:
        user_data = {
            "id": user.id,
            "fname": user.fname,
            "lname": user.lname,
            "username": user.username,
            "phoneNum": user.phoneNum,
            "email": user.email,
            "password": get_password_hash(user.password),
            "pfp": user.pfp,
            "isAuthenticated": user.isAuthenticated,
            "isDeleted": user.isDeleted
        }

        snapshot = rtdb.put_data(user_data)
        if snapshot is None:
            return {
                "status": 400,
                "msg": "User not created"
            }

        return {
            "status": 200,
            "msg": "User created"
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except TypeError as te:
        raise HTTPException(status_code=400, detail=str(te))


@app.get("/user/{username}")
async def get_user_data(username: str, data=None):
    """
    Endpoint to get user data from the Firebase Realtime Database.

    Args:
        username: Username of the target user.
        data: Data to get from the user. Defaults to None if needed to get all the information of user.
    """
    try:
        user_data = rtdb.get_data(username, data)

        if user_data is None:
            return {
                "status": 404,
                "msg": "User not found"
            }

        return user_data
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))


@app.put("/user/{username}")
async def update_user_data(username: str, user: User):
    """
    Endpoint to update user data in the database.

    Args:
        username: Username of the target user.
        user: User model that contains user data.
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

        snapshot = rtdb.update_data(username, user_data)
        if snapshot is None:
            return {
                "status": 404,
                "msg": "User not updated"
            }

        return {
            "status": 200,
            "msg": "User updated.",
            "content": snapshot
        }
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))


@app.put("/user/{username}/change-pass")
async def change_password(user: User):
    """
    Endpoint to change user password.

    Args:
        username: Username of the target user.
        user: User model that contains user data.
    """
    try:
        hashed_pass = rtdb.get_data(user.username, "password")

        if not verify_password(user.password, hashed_pass):
            return {
                "status": 401,
                "msg": "Incorrect password."
            }

        user_data = {
            "id": user.id,
            "fname": user.fname,
            "lname": user.lname,
            "username": user.username,
            "phoneNum": user.phoneNum,
            "email": user.email,
            "password": get_password_hash(user.password),
            "pfp": user.pfp,
            "isAuthenticated": user.isAuthenticated,
            "isDeleted": user.isDeleted
        }

        snapshot = rtdb.update_data(user.username, user_data)
        if snapshot is None:
            return {
                "status": 404,
                "msg": "User not updated"
            }

        return {
            "status": 200,
            "msg": "User updated."
        }
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))


@app.delete("/user/{username}")
def delete_user(username: str):
    """
    Endpoint to delete a user from the Firebase Realtime Database.

    Args:
        username: Username of the target
    """
    try:
        deleted = rtdb.delete_data(username)
        if not deleted:
            return {
                "status": 404,
                "msg": "User not deleted"
            }

        return {
            "status": 200,
            "msg": "User deleted"
        }
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))


# API Endpoint [Frame Operations]
@app.websocket("/ws-tracking")  # Mobile version
async def websocket_inference(websocket: WebSocket):
    """
    Endpoint for the WebSocket video feed. The server receives frames from the client that's formatted as a base64
    bytes. The server decodes the frame in a thread pool, processes the frame to get inference and annotations, and
    encodes it back to the client as bytes back.

    Args:
        websocket: WebSocket object.
    """
    await websocket.accept()
    connection_open = True
    while connection_open:
        try:
            frame_data = await websocket.receive()

            frame_byte = base64.b64decode(frame_data["text"])

            # Process frame in thread pool to avoid blocking
            (annotated_frame, features) = await asyncio.get_event_loop().run_in_executor(
                None, websocket_process_frames, frame_byte
            )

            # Encode and send result
            encoded, buffer = cv2.imencode(".jpeg", annotated_frame)

            if not encoded:
                raise WebSocketDisconnect

            return_bytes = base64.b64decode(buffer.tobytes())
            await websocket.send_bytes(return_bytes)
        except WebSocketDisconnect:
            connection_open = False
        except Exception as e:
            print(f"Error: {e}")
            connection_open = False

    if connection_open:
        await websocket.close()


# @app.get("/stream-tracking")
# async def video_feed():  # Web version
#     """
#     Endpoint for the video feed.
#     """
#     return StreamingResponse(
#         generate_frames(),
#         media_type="multipart/x-mixed-replace; boundary=frame"
#     )
