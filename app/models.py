from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
import json

Base = declarative_base()

class Modulo(Base):
    __tablename__ = "modulos"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    categories = Column(Text)  # JSON string
    status = Column(String, default="created")  # created, training, trained, error
    accuracy = Column(Float, nullable=True)
    model_path = Column(String, nullable=True)
    total_samples = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def get_categories_list(self):
        """Convierte JSON string a lista"""
        return json.loads(self.categories) if self.categories else []
    
    def set_categories_list(self, categories_list):
        """Convierte lista a JSON string"""
        self.categories = json.dumps(categories_list)

class TrainingSample(Base):
    __tablename__ = "training_samples"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    modulo_id = Column(String, nullable=False)
    category = Column(String, nullable=False)
    landmarks = Column(Text, nullable=False)  # JSON string
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    def get_landmarks_list(self):
        """Convierte JSON string a lista"""
        return json.loads(self.landmarks) if self.landmarks else []
    
    def set_landmarks_list(self, landmarks_list):
        """Convierte lista a JSON string"""
        self.landmarks = json.dumps(landmarks_list)
