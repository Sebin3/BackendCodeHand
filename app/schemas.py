from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class ModuloCreate(BaseModel):
    name: str
    categories: List[str]

class TrainingSampleCreate(BaseModel):
    category: str
    landmarks: List[float]

class ModuloResponse(BaseModel):
    id: str
    name: str
    categories: List[str]
    status: str
    accuracy: Optional[float]
    model_path: Optional[str]
    total_samples: int
    created_at: datetime
    updated_at: Optional[datetime]

class ModuloListResponse(BaseModel):
    modulos: List[ModuloResponse]
    total: int

class PredictionRequest(BaseModel):
    landmarks: List[float]

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    model_id: str

class DeleteResponse(BaseModel):
    status: str
    model_id: str
    message: str
