import re
from datetime import datetime
from typing import Optional, Union

from pydantic import BaseModel, validator


class User(BaseModel):
    id: str = datetime.strftime(datetime.now(), '%Y%H%d%m')
    fname: str
    lname: str
    username: str
    phoneNum: str
    email: str
    password: str
    pfp: Optional[str] = None
    isAuthenticated: bool = False
    isDeleted: bool = False

    @validator('password')
    def validate_password(cls, v):
        password_pattern = r'^(?=.*[A-Z])(?=.*\d)(?=.*[@$])[A-Za-z\d@$]{8,12}$'
        if not re.match(password_pattern, v):
            raise ValueError(
                "Password must be 8-12 characters long, with at least one uppercase letter, one digit, and one special character."
            )
        return v

    @validator('phoneNum')
    def validate_phone_num(cls, v):
        mobileno_pattern = r'^(?:\+63\d{10}|09\d{9})$'
        if not re.match(mobileno_pattern, v):
            raise ValueError("Invalid mobile number.")
        return v

    @validator('email')
    def validate_email(cls, v):
        email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        if not re.match(email_pattern, v):
            raise ValueError("Invalid email address.")
        return v


class LoginForm(BaseModel):
    username: str
    password: str

    @validator('username')
    def validate_username(cls, v):
        if not v:
            raise ValueError("Username cannot be empty.")
        return v

    @validator('password')
    def validate_password_not_empty(cls, v):
        if not v:
            raise ValueError("Password cannot be empty.")
        return v


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