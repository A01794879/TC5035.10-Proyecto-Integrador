# ---- Etapa 1: Construcción de dependencias (builders stage) ----
    FROM python:3.11-buster as builder

    WORKDIR /app
    
    COPY requirements.txt .
    
    # Instala dependencias en un directorio aislado
    RUN pip install --no-cache-dir --upgrade -r requirements.txt --target /install
    
    # ---- Etapa 2: Imagen final de ejecución (runtime stage) ----
    FROM python:3.11-slim-buster
    
    # Instala dependencias de tiempo de ejecución necesarias para numpy/faiss
    RUN apt-get update && apt-get install -y \
        libgfortran5 \
        libblas3 \
        liblapack3 \
        && rm -rf /var/lib/apt/lists/*
    
    WORKDIR /app
    
    # Copia las dependencias instaladas de la etapa 'builder' al entorno final
    # ¡Importante: la ruta del sitio-packages también cambia con la versión de Python!
    COPY --from=builder /install /usr/local/lib/python3.11/site-packages/
    COPY --from=builder /usr/local/bin/ /usr/local/bin/
    
    # Copia tu aplicación Flask
    COPY . .
    
    # Expone el puerto que usa tu aplicación
    EXPOSE 8080
    
    # Comando para ejecutar la aplicación
    CMD ["python", "main.py"]