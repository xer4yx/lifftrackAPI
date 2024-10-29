from lifttrack.dbhandler.rtdbHelper import rtdb
from lifttrack.models import User, Token, AppInfo, LoginForm
from lifttrack.comvis import cv2, websocket_process_frames

from lifttrack import timedelta, threading, asyncio
from lifttrack.auth import (create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES,
                            get_current_user, verify_password, get_password_hash)

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
    """
    Root endpoint.
    """
    return {"msg": "Welcome to LiftTrack!"}


# API Endpoint [About App]
@app.get("/app_info")
async def get_app_info(appinfo: AppInfo):
    """
    Endpoint to get information about the app.
    """
    return await appinfo


# API Endpoint [Authentication Operations]
@app.post("/login")
async def login(login_form: LoginForm):
    """
    Endpoint to login a user.
    """
    user_data = rtdb.get_data(login_form.username)

    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")

    if not verify_password(login_form.password, user_data["password"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    return {"message": "Login successful", "success": True}


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
            "password": get_password_hash(user.password),
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


# API Endpoint [Frame Operations]
# TODO: Implement roboflow inference and 3D CNN Inference for Web and Mobile versions
@app.websocket("/ws-tracking")  # Mobile version
async def websocket_inference(websocket: WebSocket):
    await websocket.accept()
    connection_open = True
    while connection_open:
        try:
            frame_data = await websocket.receive()

            print(frame_data)

            # if "bytes" in message:
            #     frame_data = message["bytes"]
            # elif "text" in message:
            #     # Handle text messages if needed
            #     print(f"Received text message: {message['text']}")
            #     continue
            # else:
            #     print("Unsupported message type")
            #     continue

            # Process frame in thread pool to avoid blocking
            annotated_frame = await asyncio.get_event_loop().run_in_executor(
                None, websocket_process_frames, frame_data
            )

            # Encode and send result
            encoded, buffer = cv2.imencode('.jpeg', annotated_frame)

            if not encoded:
                print("Error encoding frame")
                raise WebSocketDisconnect

            await websocket.send_bytes(buffer.tobytes())
            print("Frame sent")
        except WebSocketDisconnect:
            connection_open = False
        except Exception as e:
            print(f"Error: {str(e)}")
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
