# Evaluación Parcial 1 - Machine Learning
**Alumno:** Vicente Alarcón

## Descripción del Proyecto
Este proyecto es una evaluación práctica donde aplicamos conceptos de Machine Learning para analizar y predecir datos de una tienda. Utilizamos el framework **Kedro** para ordenar nuestro código en diferentes pasos (pipelines), desde la limpieza de los datos crudos hasta el entrenamiento de los modelos.

### Objetivos de los Modelos
Hemos entrenado tres modelos diferentes para resolver distintos problemas con nuestros datos:
1. **Regresión:** Queremos predecir el `monto_total_venta` de una transacción. *(Nota: Tuvimos que sacar las variables `cantidad` y `precio_unitario` de este modelo porque estaban causando "fuga de datos" o Data Leakage, haciendo que el modelo simplemente multiplicara los valores y diera un resultado irrealmente perfecto).*
2. **Clasificación:** Buscamos predecir a qué `segmento` pertenece un cliente (ej. Nuevo, Premium) basándonos en sus datos.
3. **Clustering:** Usamos K-Means para agrupar a los clientes y encontrar patrones ocultos en las ventas.

---

## Cómo ejecutar este proyecto

Este proyecto está configurado para Python 3.12.7. Para hacerlo funcionar en tu computadora, sigue estos pasos:

1. Crea un entorno virtual y actívalo:
```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Instala las librerías necesarias:
```bash
pip install -r requirements.txt
```

3. Ejecuta todo el flujo de datos (pipelines):
```bash
kedro run
```

Al terminar, los modelos entrenados quedarán guardados en la carpeta `data/06_models/` y los reportes de rendimiento en `data/08_reporting/`.

---

## Estructura de nuestros Pipelines

El proyecto está dividido en los siguientes pasos lógicos:

* **1. data_ingestion:** Solo carga los archivos `.csv` originales (ventas, productos, clientes, devoluciones) usando el Data Catalog de Kedro.
* **2. data_cleaning:** Limpiamos los datos. Aquí rellenamos los valores nulos (n/a) y arreglamos los valores atípicos (outliers). Para los outliers decidimos usar "Winzorización" en lugar de borrarlos, así no perdemos datos valiosos de clientes reales que gastaron mucho.
* **3. data_integration:** Juntamos todas las tablas en una sola. También borramos las columnas que no sirven para predecir (como los IDs o correos) y escalamos los números usando `StandardScaler` para que los modelos no se confundan con variables de distintos tamaños.
* **4. ml_modeling:** Usamos la librería TPOT para que pruebe varios modelos automáticamente y elija el mejor para regresión y clasificación. Luego guardamos los resultados en formato `.pkl`.

---

## Justificación de Métricas obtenidas

**Modelo de Clasificación (Accuracy bajo):** 
Si revisan los reportes generados, notarán que el modelo de clasificación obtuvo un Accuracy cercano al 42%. Aunque parece bajo, esto tiene una explicación técnica. Después de cruzar todas las tablas y limpiar los datos faltantes, solo nos quedaron cerca de 33 filas válidas con información en la columna `segmento`. Como tenemos 4 segmentos distintos, el algoritmo intenta aprender con casi 8 registros por clase, lo cual es muy poco para Machine Learning. Un resultado mayor en este escenario significaría que el modelo está sobreajustado (*Overfitting*).

**Modelo de Regresión (Fuga de Datos corregida):**
Al principio nuestro modelo de regresión tenía un $R^2$ de casi 1.0 (0.999). Nos dimos cuenta de que le estábamos pasando `cantidad` y `precio_unitario` como variables de entrada, y como el monto total es simplemente `cantidad * precio_unitario`, el modelo solo aprendió a hacer esa multiplicación matemática (Data Leakage). Para solucionarlo, excluimos esas variables en el archivo `conf/base/parameters.yml`, forzando al modelo a predecir usando otras variables reales como canales de venta y stock.

---
En la carpeta `notebooks/01_EDA_Profesional.ipynb` incluí un cuaderno exploratorio donde analizo gráficamente algunos de estos puntos, como la diferencia al tratar los outliers.
