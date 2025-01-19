from typing import Final
from core.entities import User

# Custom Headers
HEADER_TRACE_ID: Final[str] = "X-Trace-ID"
HEADER_REQUEST_ID: Final[str] = "X-Request-ID"
HEADER_API_VERSION: Final[str] = "X-API-Version"

# Response Messages
MSG_INVALID_CREDENTIALS: Final[str] = "Invalid credentials provided"
MSG_UNAUTHORIZED: Final[str] = "Unauthorized access"
MSG_NOT_FOUND: Final[str] = "Resource not found"
MSG_VALIDATION_ERROR: Final[str] = "Validation error occurred"

# Error Codes
ERR_AUTH_INVALID: Final[str] = "AUTH_001"
ERR_AUTH_EXPIRED: Final[str] = "AUTH_002"
ERR_VALIDATION: Final[str] = "VAL_001"
ERR_NOT_FOUND: Final[str] = "NF_001"

# User Router Responses
RESPONSE_200: Final[str] = {
    "description": "OK",
    "content": {
        "application/json": {
            "example": {
                "msg": "OK"
            }
        }
    }
}
RESPONSE_201: Final[str] = {
    "description": "Created",
    "content": {
        "application/json": {
            "example": {
                "msg": "User created."
            }
        }
    }
}
RESPONSE_400: Final[str] = {
    "description": "Bad Request",
    "content": {
        "application/json": {
            "example": {
                "msg": "User creation failed."
            }
        }
    }
}
RESPONSE_401: Final[str] = {
    "description": "Unauthorized",
    "content": {
        "application/json": {
            "example": {
                "msg": "Incorrect password."
            }
        }
    }
}
RESPONSE_404: Final[str] = {
    "description": "Not found",
    "content": {
        "application/json": {
            "example": {
                "msg": "User not found."
            }
        }
    }
}
RESPONSE_405: Final[str] = {
    "description": "Method not allowed",
    "content": {
        "application/json": {
            "example": {
                "msg": "Method not allowed."
            }
        }
    }
}
RESPONSE_412: Final[str] = {
    "description": "Precondition Failed",
    "content": {
        "application/json": {
            "example": {
                "msg": "Precondition Failed."
            }
        }
    }
}
RESPONSE_422: Final[str] = {
    "description": "Unprocessable Entity",
    "content": {
        "application/json": {
            "example": {
                "msg": "Unprocessable Entity."
            }
        }
    }
}
RESPONSE_500: Final[str] = {
    "description": "Internal Server Error",
    "content": {
        "application/json": {
            "example": {
                "msg": "Internal Server Error."
            }
        }
    }
}
RESPONSE_503: Final[str] = {
    "description": "Service Unavailable",
    "content": {
        "application/json": {
            "example": {
                "msg": "Service Unavailable."
            }
        }
    }
}
