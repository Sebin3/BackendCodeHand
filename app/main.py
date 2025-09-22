from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from typing import List
import uuid
import os
import json
import joblib
import numpy as np
import csv
import io
from datetime import datetime

from .database import get_db, init_database
from .models import Modulo, TrainingSample
from .schemas import (
    ModuloCreate, TrainingSampleCreate, ModuloResponse,
    PredictionRequest, PredictionResponse, DeleteResponse
)

app = FastAPI(
    title="ASL Recognition API",
    description="API para reconocimiento dinámico de lenguaje de señas",
    version="2.0.0"
)

# Inicializar base de datos al iniciar
@app.on_event("startup")
async def startup_event():
    init_database()

# CORS
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoints

@app.get("/")
def read_root():
    return {
        "message": "API de reconocimiento de señas activa",
        "version": "2.0.0",
        "endpoints": {
            "models": "/models - GET para listar modelos",
            "create": "/models/create - POST para crear modelo",
            "capture": "/models/{id}/capture - POST para capturar datos",
            "train": "/models/{id}/train - POST para entrenar modelo",
            "predict": "/models/{id}/predict - POST para predecir",
            "delete": "/models/{id} - DELETE para eliminar modelo",
            "import": "/models/{id}/import - POST para importar dataset JSON",
            "import_csv": "/models/{id}/import-csv - POST para importar dataset CSV"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "ok", "database": "connected"}

@app.get("/models", response_model=List[ModuloResponse])
def list_models(db: Session = Depends(get_db)):
    """Listar todos los módulos"""
    modulos = db.query(Modulo).all()
    
    # Convertir objetos de la BD al formato correcto
    result = []
    for modulo in modulos:
        result.append(ModuloResponse(
            id=modulo.id,
            name=modulo.name,
            categories=modulo.get_categories_list(),  # Convertir JSON string a lista
            status=modulo.status,
            accuracy=modulo.accuracy,
            model_path=modulo.model_path,
            total_samples=modulo.total_samples,
            created_at=modulo.created_at,
            updated_at=modulo.updated_at
        ))
    
    return result

@app.get("/models/{model_id}/samples/{category}")
def get_category_samples(model_id: str, category: str, db: Session = Depends(get_db)):
    """Obtener muestras de una categoría específica"""
    # Verificar que el módulo existe
    modulo = db.query(Modulo).filter(Modulo.id == model_id).first()
    if not modulo:
        raise HTTPException(status_code=404, detail="Módulo no encontrado")
    
    # Contar muestras de la categoría
    count = db.query(TrainingSample).filter(
        TrainingSample.modulo_id == model_id,
        TrainingSample.category == category
    ).count()
    
    return {
        "model_id": model_id,
        "category": category,
        "count": count
    }

@app.post("/models/create")
async def create_model(request: ModuloCreate, db: Session = Depends(get_db)):
    """Crear nuevo módulo"""
    model_id = str(uuid.uuid4())
    
    modulo = Modulo(
        id=model_id,
        name=request.name,
        categories=json.dumps(request.categories),
        status="created"
    )
    
    db.add(modulo)
    db.commit()
    db.refresh(modulo)
    
    return {
        "model_id": model_id,
        "status": "created",
        "message": f"Módulo '{request.name}' creado correctamente"
    }

@app.post("/models/{model_id}/capture")
async def capture_sample(
    model_id: str,
    request: TrainingSampleCreate,
    db: Session = Depends(get_db)
):
    """Capturar muestra de entrenamiento"""
    # Verificar que el módulo existe
    modulo = db.query(Modulo).filter(Modulo.id == model_id).first()
    if not modulo:
        raise HTTPException(status_code=404, detail="Módulo no encontrado")
    
    # Crear muestra
    sample = TrainingSample(
        modulo_id=model_id,
        category=request.category,
        landmarks=json.dumps(request.landmarks)
    )
    
    db.add(sample)
    db.commit()
    
    # Actualizar contador de muestras
    total_samples = db.query(TrainingSample).filter(TrainingSample.modulo_id == model_id).count()
    modulo.total_samples = total_samples
    db.commit()
    
    return {
        "status": "captured",
        "total_samples": total_samples,
        "category_samples": db.query(TrainingSample).filter(
            TrainingSample.modulo_id == model_id,
            TrainingSample.category == request.category
        ).count()
    }

@app.post("/models/{model_id}/train")
async def train_model(model_id: str, db: Session = Depends(get_db)):
    """Entrenar modelo"""
    # Verificar que el módulo existe
    modulo = db.query(Modulo).filter(Modulo.id == model_id).first()
    if not modulo:
        raise HTTPException(status_code=404, detail="Módulo no encontrado")
    
    # Obtener datos de entrenamiento
    samples = db.query(TrainingSample).filter(TrainingSample.modulo_id == model_id).all()
    if not samples:
        raise HTTPException(status_code=400, detail="No hay datos para entrenar")
    
    try:
        # Preparar datos
        X = [sample.get_landmarks_list() for sample in samples]
        y = [sample.category for sample in samples]
        
        # Entrenar modelo
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Calcular precisión
        accuracy = model.score(X_test, y_test)
        
        # Guardar modelo en directorio persistente
        model_dir = os.getenv("MODEL_DIR", "./models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = f"{model_dir}/{model_id}.pkl"
        joblib.dump(model, model_path)
        
        # Actualizar módulo
        modulo.status = "trained"
        modulo.accuracy = accuracy
        modulo.model_path = model_path
        db.commit()
        
        return {
            "status": "trained",
            "accuracy": accuracy,
            "total_samples": len(samples),
            "model_path": model_path
        }
        
    except Exception as e:
        modulo.status = "error"
        db.commit()
        raise HTTPException(status_code=500, detail=f"Error entrenando modelo: {str(e)}")

@app.post("/models/{model_id}/predict", response_model=PredictionResponse)
async def predict(
    model_id: str,
    request: PredictionRequest,
    db: Session = Depends(get_db)
):
    """Predecir usando modelo entrenado"""
    print(f"🔍 [PREDICT] Iniciando predicción para modelo: {model_id}")
    print(f"📊 [PREDICT] Landmarks recibidos: {len(request.landmarks)} elementos")
    print(f"📊 [PREDICT] Primeros 5 landmarks: {request.landmarks[:5]}")
    
    # Verificar que el módulo existe y está entrenado
    modulo = db.query(Modulo).filter(Modulo.id == model_id).first()
    if not modulo:
        print(f"❌ [PREDICT] Módulo no encontrado: {model_id}")
        raise HTTPException(status_code=404, detail="Módulo no encontrado")
    
    print(f"✅ [PREDICT] Módulo encontrado: {modulo.name}")
    print(f"📋 [PREDICT] Status: {modulo.status}")
    print(f"📋 [PREDICT] Model path: {modulo.model_path}")
    
    if modulo.status != "trained":
        print(f"❌ [PREDICT] Módulo no está entrenado: {modulo.status}")
        raise HTTPException(status_code=400, detail="Módulo no está entrenado")
    
    if not modulo.model_path:
        print(f"❌ [PREDICT] Model path es None")
        raise HTTPException(status_code=500, detail="Modelo no encontrado en disco")
    
    if not os.path.exists(modulo.model_path):
        print(f"❌ [PREDICT] Archivo modelo no existe: {modulo.model_path}")
        raise HTTPException(status_code=500, detail="Modelo no encontrado en disco")
    
    print(f"✅ [PREDICT] Archivo modelo existe: {modulo.model_path}")
    
    try:
        print(f"🔄 [PREDICT] Cargando modelo...")
        # Cargar modelo
        model = joblib.load(modulo.model_path)
        print(f"✅ [PREDICT] Modelo cargado exitosamente")
        
        print(f"🔄 [PREDICT] Preparando datos para predicción...")
        # Predecir
        arr = np.array(request.landmarks).reshape(1, -1)
        print(f"📊 [PREDICT] Array shape: {arr.shape}")
        print(f"📊 [PREDICT] Primeros 10 valores: {arr[0][:10]}")
        
        print(f"🔄 [PREDICT] Ejecutando predicción...")
        prediction = model.predict(arr)[0]
        print(f"✅ [PREDICT] Predicción: {prediction}")
        
        print(f"🔄 [PREDICT] Calculando confianza...")
        confidence = float(max(model.predict_proba(arr)[0]))
        print(f"✅ [PREDICT] Confianza: {confidence}")
        
        result = PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            model_id=model_id
        )
        
        print(f"🎉 [PREDICT] Respuesta exitosa: {result}")
        return result
        
    except Exception as e:
        print(f"❌ [PREDICT] Error en predicción: {str(e)}")
        print(f"❌ [PREDICT] Tipo de error: {type(e).__name__}")
        import traceback
        print(f"❌ [PREDICT] Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

@app.delete("/models/{model_id}", response_model=DeleteResponse)
async def delete_model(model_id: str, db: Session = Depends(get_db)):
    """Eliminar módulo y todos sus datos"""
    # Verificar que el módulo existe
    modulo = db.query(Modulo).filter(Modulo.id == model_id).first()
    if not modulo:
        raise HTTPException(status_code=404, detail="Módulo no encontrado")
    
    try:
        # Eliminar modelo del disco
        if modulo.model_path and os.path.exists(modulo.model_path):
            os.remove(modulo.model_path)
        
        # Eliminar datos de entrenamiento del disco
        training_data_path = f"data/training_data/{model_id}.json"
        if os.path.exists(training_data_path):
            os.remove(training_data_path)
        
        # Eliminar muestras de entrenamiento de la base de datos
        db.query(TrainingSample).filter(TrainingSample.modulo_id == model_id).delete()
        
        # Eliminar módulo
        db.delete(modulo)
        db.commit()
        
        return DeleteResponse(
            status="deleted",
            model_id=model_id,
            message=f"Módulo '{modulo.name}' y todos sus datos eliminados correctamente"
        )
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error eliminando módulo: {str(e)}")

@app.post("/models/{model_id}/import-csv")
async def import_csv_dataset(
    model_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Importar dataset desde archivo CSV"""
    # Verificar que el módulo existe
    modulo = db.query(Modulo).filter(Modulo.id == model_id).first()
    if not modulo:
        raise HTTPException(status_code=404, detail="Módulo no encontrado")
    
    # Verificar que el archivo es CSV
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")
    
    try:
        # Leer contenido del archivo
        content = await file.read()
        csv_content = content.decode('utf-8')
        
        # Parsear CSV
        csv_reader = csv.DictReader(io.StringIO(csv_content))
        
        imported_count = 0
        batch_size = 100  # Procesar en lotes más grandes para CSV
        
        # Procesar filas del CSV
        rows = list(csv_reader)
        total_rows = len(rows)
        
        for i in range(0, total_rows, batch_size):
            batch = rows[i:i + batch_size]
            
            for row in batch:
                try:
                    # Obtener las claves de la fila
                    row_keys = list(row.keys())
                    
                    if len(row_keys) < 2:
                        continue
                    
                    # La primera columna es la categoría
                    category = row[row_keys[0]]
                    
                    # Las siguientes columnas son los landmarks
                    landmarks = []
                    for i in range(1, len(row_keys)):
                        try:
                            value = float(row[row_keys[i]])
                            landmarks.append(value)
                        except (ValueError, TypeError):
                            continue
                    
                    # Validar que tengamos landmarks válidos
                    if len(landmarks) == 0:
                        continue
                    
                    # Crear muestra
                    sample = TrainingSample(
                        modulo_id=model_id,
                        category=category,
                        landmarks=json.dumps(landmarks)
                    )
                    
                    db.add(sample)
                    imported_count += 1
                    
                except Exception as e:
                    print(f"Error procesando fila: {e}")
                    continue
            
            # Commit del lote
            db.commit()
            print(f"Lote procesado: {imported_count} muestras importadas")
        
        # Actualizar contador de muestras del módulo
        total_samples = db.query(TrainingSample).filter(TrainingSample.modulo_id == model_id).count()
        modulo.total_samples = total_samples
        db.commit()
        
        return {
            "status": "imported",
            "imported_count": imported_count,
            "total_samples": total_samples,
            "message": f"Dataset CSV importado: {imported_count} muestras agregadas"
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error importando CSV: {str(e)}")