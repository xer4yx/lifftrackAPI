from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pydantic import EmailStr
from pydantic_extra_types.phone_numbers import PhoneNumber, PhoneNumberValidator
from typing import Annotated, Any, Dict, Optional, Union
from .base import EntityBase, EntityDefaultBase

PHNumberType = Annotated[
    Union[str, PhoneNumber], 
    PhoneNumberValidator(
        default_region='PH', 
        supported_regions=['PH']
    )
]

@dataclass
class UserPhoneNumber(PhoneNumber):
    """User phone number entity"""
    default_region_code = 'PH'
    supported_regions = ['PH', 'US']
    phone_format = "E164"
    
    def __init__(self, phone_number: str):
        super().__init__(phone_number)
        
    def __str__(self) -> str:
        return f"+63{self.phone_number}"

@dataclass
class UserBase(EntityBase):
    """Base class for all user entities"""
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    password: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone_number: Optional[UserPhoneNumber] = None
    
    @property
    def full_number(self) -> str:
        """Get user's full number"""
        return f"+63{self.phone_number}"
    
@dataclass
class UserDefaultBase(EntityDefaultBase):
    """Default values for user entities"""
    profile_picture: Optional[str] = None
    is_authenticated: Optional[bool] = False
    is_deleted: Optional[bool] = False
    last_login: Optional[datetime] = None
    
    def update_last_login(self) -> None:
        """Update last login timestamp"""
        self.last_login = datetime.now(timezone.utc).isoformat()

@dataclass
class User(UserDefaultBase, UserBase):
    """User entity representing a system user"""
    @property
    def full_name(self) -> str:
        """Get user's full name"""
        return f"{self.first_name} {self.last_name}"
    
    def set_authenticated(self) -> None:
        """Mark user as authenticated"""
        self.is_authenticated = True
        self.last_login = datetime.now(timezone.utc).isoformat()
        self.update_timestamp()
    
    def update_profile(
        self,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        email: Optional[str] = None,
        phone_number: Optional[str] = None,
        profile_picture: Optional[str] = None
    ) -> None:
        """Update user profile information"""
        if first_name:
            self.first_name = first_name
        if last_name:
            self.last_name = last_name
        if email:
            self.email = email
        if phone_number:
            self.phone_number = phone_number
        if profile_picture:
            self.profile_picture = profile_picture
        self.update_timestamp()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert user entity to a python dictionary"""
        return asdict(self)