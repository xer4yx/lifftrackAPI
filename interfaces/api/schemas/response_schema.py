from typing import Generic, TypeVar, Optional, Any, Dict
from pydantic import BaseModel, Field
from fastapi import status

T = TypeVar('T')

class APIResponse(BaseModel, Generic[T]):
    status_code: int = Field(default=status.HTTP_200_OK)
    message: str
    data: Optional[T] = None
    metadata: Optional[Dict[str, Any]] = None

class ErrorDetail(BaseModel):
    field: Optional[str] = None
    message: str
    code: str

class ErrorResponse(BaseModel):
    status_code: int = status.HTTP_400_BAD_REQUEST
    message: str
    errors: list[ErrorDetail]
    trace_id: Optional[str] = None