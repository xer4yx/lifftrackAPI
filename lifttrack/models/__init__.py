from lifttrack import BaseModel, Union, Optional, datetime


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


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


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