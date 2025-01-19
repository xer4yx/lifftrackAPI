from typing import TypeVar, Optional, Any, Dict
from fastapi import Response, Header, status
from interfaces.api.schemas.response_schema import (
    APIResponse, 
    ErrorResponse, 
    ErrorDetail
)
from interfaces.api.constants import HEADER_TRACE_ID
import uuid

T = TypeVar('T')

def success_response(
    data: Optional[T] = None,
    message: str = "Operation successful",
    metadata: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None
) -> APIResponse[T]:
    return APIResponse(
        status=status.HTTP_200_OK,
        message=message,
        data=data,
        metadata=metadata
    )

def error_response(
    message: str,
    errors: list[ErrorDetail],
    headers: Optional[Dict[str, str]] = None
) -> ErrorResponse:
    return ErrorResponse(
        message=message,
        errors=errors,
        trace_id=str(uuid.uuid4())
    )

def set_pagination_headers(
    response: Response,
    total: int,
    page: int,
    per_page: int
) -> None:
    total_pages = (total + per_page - 1) // per_page
    response.headers["X-Total-Count"] = str(total)
    response.headers["X-Page"] = str(page)
    response.headers["X-Per-Page"] = str(per_page)
    response.headers["X-Total-Pages"] = str(total_pages)