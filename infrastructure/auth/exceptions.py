from fastapi import HTTPException, status


class AuthError(HTTPException):
    """Base class for authentication errors."""

    def __init__(
        self,
        detail: str,
        status_code: int = status.HTTP_401_UNAUTHORIZED,
        headers: dict = None,
    ):
        if headers is None:
            headers = {"WWW-Authenticate": "Bearer"}
        super().__init__(status_code=status_code, detail=detail, headers=headers)


class InvalidCredentialsError(AuthError):
    """Error raised when credentials are invalid."""

    def __init__(self, detail: str = "Invalid authentication credentials"):
        super().__init__(detail=detail)


class TokenExpiredError(AuthError):
    """Error raised when token is expired."""

    def __init__(self, detail: str = "Token has expired"):
        super().__init__(detail=detail)


class TokenInvalidError(AuthError):
    """Error raised when token is invalid."""

    def __init__(self, detail: str = "Invalid token"):
        super().__init__(detail=detail)


class TokenBlacklistedError(AuthError):
    """Error raised when token is blacklisted."""

    def __init__(self, detail: str = "Token has been invalidated"):
        super().__init__(detail=detail)


class UserNotFoundError(AuthError):
    """Error raised when user is not found."""

    def __init__(self, detail: str = "User not found"):
        super().__init__(detail=detail, status_code=status.HTTP_404_NOT_FOUND)


class ValidationError(HTTPException):
    """Error raised when input validation fails."""

    def __init__(self, detail: str = "Validation error"):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=detail
        )


class UsernameExistsError(ValidationError):
    """Error raised when username already exists."""

    def __init__(self, detail: str = "Username already exists"):
        super().__init__(detail=detail)


class InvalidPasswordError(ValidationError):
    """Error raised when password does not meet requirements."""

    def __init__(self, detail: str = "Password does not meet security requirements"):
        super().__init__(detail=detail)
