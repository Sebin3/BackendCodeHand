# ASL Recognition Backend

API para reconocimiento dinámico de lenguaje de señas americano (ASL).

## Estructura del Proyecto

```
├── app/
│   ├── main.py              # FastAPI app principal
│   ├── database.py          # Configuración de base de datos
│   ├── models.py            # Modelos SQLAlchemy
│   └── schemas.py           # Esquemas Pydantic
├── data/
│   ├── models/              # Modelos entrenados (.pkl)
│   └── training_data/       # Datos de entrenamiento (.json)
├── requirements.txt
├── Dockerfile
├── render.yaml
└── README.md
```

## Instalación

1. Instalar dependencias:
```bash
pip install -r requirements.txt
```

2. Ejecutar la aplicación:
```bash
uvicorn app.main:app --reload
```

## Endpoints

- `GET /` - Información de la API
- `GET /health` - Health check
- `GET /models` - Listar todos los módulos
- `POST /models/create` - Crear nuevo módulo
- `POST /models/{id}/capture` - Capturar muestra de entrenamiento
- `POST /models/{id}/train` - Entrenar modelo
- `POST /models/{id}/predict` - Predecir con modelo
- `DELETE /models/{id}` - Eliminar módulo

## Despliegue

Para desplegar en Render, simplemente conecta tu repositorio y Render detectará automáticamente el Dockerfile.
