from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Base
import os

# Configuración de base de datos
DATABASE_URL = os.getenv("DATABASE_URL") or os.getenv("database", "sqlite:///./data/asl_recognition.db")

# Crear motor de base de datos
if "postgresql" in DATABASE_URL:
    # PostgreSQL
    engine = create_engine(DATABASE_URL)
else:
    # SQLite (desarrollo local)
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False}
    )

# Crear sesión
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """Dependencia para obtener sesión de base de datos"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    """Crear todas las tablas"""
    Base.metadata.create_all(bind=engine)

def init_database():
    """Inicializar base de datos"""
    create_tables()
    print("Base de datos inicializada correctamente")
