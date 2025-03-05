from pydantic import BaseModel
from typing import List, Dict, Any


class ClassifierRequest(BaseModel):
    DataSet: str
    data: List[Dict[str, Any]]



    