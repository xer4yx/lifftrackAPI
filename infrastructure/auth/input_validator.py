from typing import Any, Dict, Annotated
import re
from core.interfaces import InputValidator, DatabaseRepository
from infrastructure import get_admin_firebase_db
from utilities.monitoring import MonitoringFactory

class DataValidator(InputValidator):
    def __init__(self, db: DatabaseRepository = Annotated[DatabaseRepository, get_admin_firebase_db]):
        self.logger = MonitoringFactory.get_logger(
            module_name="input-validator",
            log_dir="logs/infrastructure/auth"
        )
        self._validation_errors = []
        users = db.query(path="users", order_by="username")
        self._existing_usernames = set(user.get('username') for user in users if user.get('username'))
    
    def validate(self, data: Dict[str, Any]) -> bool:
        """Validates user input and returns True if valid, False otherwise."""
        self._validation_errors = []  # Reset errors before validation
        
        password_pattern = r'^(?=.*[A-Z])(?=.*\d)(?=.*[@$])[A-Za-z\d@$]{8,12}$'
        mobileno_pattern = r'^(?:\+63\d{10}|09\d{9})$'
        email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        
        # Extract values from dictionary
        username = data.get('username')
        password = data.get('password')
        phone_number = data.get('phone_number')
        email = data.get('email')
        
        if not username or not password or not phone_number or not email:
            self._validation_errors.append("Missing required fields in user data")
            return False
        
        # Check for existing username
        if username in self._existing_usernames:
            self._validation_errors.append(f"Username '{username}' already exists")
            return False

        # Validate password format
        if not re.match(password_pattern, password):
            self._validation_errors.append(
                "Password must be 8-12 characters long and contain at least "
                "one uppercase letter, one number, and one special character (@$)"
            )
            return False

        # Validate phone number format
        if not re.match(mobileno_pattern, phone_number):
            self._validation_errors.append(
                "Phone number must be in format: +63XXXXXXXXXX or 09XXXXXXXXX"
            )
            return False

        # Validate email format
        if not re.match(email_pattern, email):
            self._validation_errors.append("Invalid email format")
            return False

        self._logger.info(f"User data validated successfully: {username}")
        return True
    
    def get_validation_errors(self) -> list:
        """Returns the list of validation errors from the last validation attempt."""
        return self._validation_errors