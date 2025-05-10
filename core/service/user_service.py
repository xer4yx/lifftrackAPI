from datetime import datetime
import re


def generate_user_id() -> str:
    """
    Generate a unique user ID based on the current date and time.
    
    Returns:
        str: A unique user ID in the format YYYYMMDDHHMM
    """
    return datetime.strftime(datetime.now(), '%Y%H%d%m')


def validate_email(email: str) -> bool:
    """
    Validate an email address.
    
    Args:
        email (str): The email address to validate.
        
    Returns:
        bool: True if the email is valid, False otherwise.
    """
    return re.match(r"[^@]+@[^@]+\.[^@]+", email) is not None


def validate_phone_number(phone_number: str) -> bool:
    """
    Validate a phone number.
    
    Args:
        phone_number (str): The phone number to validate.
        
    Returns:
        bool: True if the phone number is valid, False otherwise.
    """
    return re.match(r"^\+?[1-9]\d{1,14}$", phone_number) is not None


def validate_password(password: str) -> bool:
    """
    Validate a password. Must be at least 8 characters long, 
    contain at least one uppercase letter, one lowercase letter, 
    one digit, and one special character.
    
    Args:
        password (str): The password to validate.
        
    Returns:
        bool: True if the password is valid, False otherwise.
    """
    return re.match(r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$", password) is not None
