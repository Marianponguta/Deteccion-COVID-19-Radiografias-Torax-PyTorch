# Detección de COVID-19 con Radiografías de Tórax usando PyTorch

Sistema de clasificación de imágenes de radiografías de tórax utilizando Deep Learning para detectar COVID-19, Neumonía Viral y casos normales mediante Transfer Learning con ResNet18.

## Descripción del Proyecto

Este proyecto implementa un modelo de clasificación de imágenes médicas basado en redes neuronales convolucionales (CNN) para identificar automáticamente tres clases de radiografías de tórax:
- **Normal**: Radiografías sin patologías
- **Viral Pneumonia**: Neumonía de origen viral
- **COVID-19**: Casos positivos de COVID-19

El modelo utiliza Transfer Learning con la arquitectura **ResNet18** preentrenada en ImageNet y ajustada específicamente para esta tarea de clasificación médica. El proyecto incluye técnicas de balanceo de clases mediante `WeightedRandomSampler` para manejar el desbalance en el dataset.

## Características Principales

### Procesamiento de Datos
- **Balanceo de clases**: Implementación de `WeightedRandomSampler` para equilibrar la representación de cada clase durante el entrenamiento
- **Transformaciones de imágenes**: 
  - Redimensionamiento a 224x224 píxeles
  - Normalización con valores de ImageNet
  - Data augmentation (flip horizontal aleatorio en entrenamiento)
- **Split de datos**: División automática en conjuntos de entrenamiento y prueba (40 imágenes por clase para test)

### Modelo y Entrenamiento
- **Arquitectura**: ResNet18 preentrenada con capa final modificada para 3 clases
- **Optimización**: Adam optimizer con learning rate de 3e-5
- **Función de pérdida**: CrossEntropyLoss
- **Early stopping**: Detención automática al alcanzar 94% de accuracy

### Visualización y Análisis
- **Distribución del dataset**: Gráficos de barras y pie charts mostrando la distribución de clases
- **Balanceo visual**: Comparación antes/después del balanceo de clases
- **Predicciones**: Visualización de imágenes con predicciones (correctas en verde, incorrectas en rojo)
- **Métricas de evaluación**:
  - Matriz de confusión (absoluta y normalizada)
  - Curvas ROC por clase con AUC
  - Precision, F1-Score y Accuracy por clase
  - Promedios micro, macro y weighted

## Dataset

El proyecto utiliza el **COVID-19 Radiography Database** disponible en Kaggle:
- **Fuente**: [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
- **Contenido**: 
  - COVID-19: Imágenes de casos positivos
  - Normal: Radiografías sin patologías
  - Viral Pneumonia: Casos de neumonía viral
- **Formato**: Imágenes PNG de 299x299 píxeles

### Estructura del Dataset
```
COVID-19 Radiography Database/
├── README.md
├── Normal/           # Imágenes de casos normales
├── Viral Pneumonia/  # Imágenes de neumonía viral
├── Covid 19/         # Imágenes de COVID-19
└── test/             # Conjunto de prueba (generado automáticamente)
    ├── Normal/
    ├── Viral Pneumonia/
    └── Covid 19/
```

**Nota**: El dataset NO está incluido en este repositorio. Debe descargarse desde Kaggle siguiendo las instrucciones de instalación.

## Requisitos del Sistema

- **Python**: 3.10.11
- **Sistema Operativo**: Windows (instrucciones específicas para PowerShell)
- **Memoria RAM**: Mínimo 8 GB recomendado
- **GPU**: Opcional pero recomendada para entrenamiento más rápido

## Instalación y Configuración

### 1. Clonar el Repositorio

```powershell
git clone <url-del-repositorio>
cd "Detecting COVID-19 with Chest X-Ray using PyTorch"
```

### 2. Instalar Pipenv

Si no tienes Pipenv instalado:

```powershell
pip install pipenv
```

### 3. Crear el Entorno Virtual e Instalar Dependencias

```powershell
pipenv install
```

Este comando creará un entorno virtual e instalará todas las dependencias especificadas en el `Pipfile`:
- matplotlib
- torch
- torchvision
- numpy
- pillow
- ipython
- seaborn
- plotly
- nbformat
- scikit-learn
- pandas

### 4. Descargar el Dataset

1. Accede a [COVID-19 Radiography Database en Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
2. Descarga el dataset
3. Extrae el contenido en la raíz del proyecto
4. Asegúrate de que la estructura de carpetas sea:
   ```
   COVID-19 Radiography Database/
   ├── Normal/
   ├── Viral Pneumonia/
   └── Covid 19/
   ```

### 5. Activar el Entorno Virtual

```powershell
pipenv shell
```

## Ejecución del Proyecto

### Opción 1: Jupyter Notebook (Recomendado)

1. Inicia Jupyter Notebook:
   ```powershell
   pipenv run jupyter notebook
   ```

2. Abre el archivo `Con Balanceo.ipynb` en el navegador

3. Ejecuta las celdas en orden:
   - **Importación de librerías**: Carga todas las dependencias necesarias
   - **Análisis exploratorio**: Visualiza la distribución del dataset
   - **Preparación de datos**: Crea los conjuntos de entrenamiento y test
   - **Transformaciones**: Aplica preprocesamiento a las imágenes
   - **Balanceo**: Implementa el balanceo de clases
   - **Visualización**: Muestra ejemplos de imágenes
   - **Creación del modelo**: Configura ResNet18
   - **Entrenamiento**: Entrena el modelo (se detendrá automáticamente al alcanzar 94% accuracy)
   - **Evaluación**: Genera métricas y visualizaciones finales

### Opción 2: VS Code con Jupyter

1. Abre VS Code en la carpeta del proyecto
2. Asegúrate de tener la extensión de Jupyter instalada
3. Abre `Con Balanceo.ipynb`
4. Selecciona el kernel de Python del entorno Pipenv
5. Ejecuta las celdas secuencialmente

### Opción 3: Ejecutar desde Python

```powershell
pipenv run python -c "import nbformat; from nbconvert import PythonExporter; nb = nbformat.read('Con Balanceo.ipynb', as_version=4); code = PythonExporter().from_notebook_node(nb)[0]; exec(code)"
```

## Estructura del Proyecto

```
.
├── Con Balanceo.ipynb           # Notebook principal con todo el pipeline
├── Pipfile                      # Dependencias del proyecto
├── Pipfile.lock                 # Versiones bloqueadas de dependencias
├── newplot.png                  # Gráfico de distribución del dataset
├── README.md                    # Este archivo
├── LICENSE                      # Licencia del proyecto
├── .gitignore                   # Archivos ignorados por Git
└── COVID-19 Radiography Database/  # Dataset (no incluido en el repo)
```

## Resultados Esperados

El modelo está diseñado para alcanzar:
- **Accuracy**: ≥ 94% en el conjunto de prueba
- **Precision y F1-Score**: Métricas altas para las tres clases
- **AUC-ROC**: Valores cercanos a 1.0 para cada clase

Las visualizaciones generadas incluyen:
- Distribución del dataset antes y después del balanceo
- Ejemplos de predicciones con etiquetas correctas/incorrectas
- Matriz de confusión normalizada y absoluta
- Curvas ROC individuales para cada clase
- Tabla de métricas detalladas (Precision, F1, Accuracy)

## Notas Técnicas

### Balanceo de Clases
El proyecto implementa balanceo de clases mediante `WeightedRandomSampler` de PyTorch, que ajusta la probabilidad de muestreo de cada clase según su frecuencia inversa. Esto asegura que el modelo vea aproximadamente la misma cantidad de ejemplos de cada clase durante el entrenamiento, evitando sesgos hacia la clase mayoritaria.

### Transfer Learning
Se utiliza ResNet18 preentrenada en ImageNet con pesos congelados en las capas convolucionales y solamente la capa final (fully connected) ajustada para 3 clases. Esto permite aprovechar las características visuales aprendidas en millones de imágenes y adaptarlas a radiografías médicas.

### Early Stopping
El entrenamiento se detiene automáticamente cuando la accuracy en validación alcanza el 94%, evitando sobreentrenamiento y ahorrando tiempo de cómputo.

## Créditos

- **Notebook base**: Basado en el proyecto guiado [Detecting COVID-19 with Chest X Ray using PyTorch](https://www.coursera.org/projects/covid-19-detection-x-ray) en Coursera
- **Dataset**: [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) en Kaggle

### Citas del Dataset

Si utilizas este dataset, por favor cita los siguientes artículos:

1. M.E.H. Chowdhury, T. Rahman, A. Khandakar, R. Mazhar, M.A. Kadir, Z.B. Mahbub, K.R. Islam, M.S. Khan, A. Iqbal, N. Al-Emadi, M.B.I. Reaz, M. T. Islam, "Can AI help in screening Viral and COVID-19 pneumonia?" IEEE Access, Vol. 8, 2020, pp. 132665 - 132676.

2. Rahman, T., Khandakar, A., Qiblawey, Y., Tahir, A., Kiranyaz, S., Kashem, S.B.A., Islam, M.T., Maadeed, S.A., Zughaier, S.M., Khan, M.S. and Chowdhury, M.E., 2020. Exploring the Effect of Image Enhancement Techniques on COVID-19 Detection using Chest X-ray Images. arXiv preprint arXiv:2012.02238.

## Licencia

Copyright © 2025. Todos los derechos reservados.

Este software está protegido por derechos de autor. No se permite el uso comercial ni la redistribución sin permiso explícito del autor. Consulta el archivo `LICENSE` para más detalles.
