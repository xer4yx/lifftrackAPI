from pydantic import BaseModel
from typing import Union


class OutboundData(BaseModel):
    type: str
    data: Union[bytes, str]

