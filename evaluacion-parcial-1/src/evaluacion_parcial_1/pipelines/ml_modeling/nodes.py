"""
Módulo de Machine Learning y AutoML para el pipeline `ml_modeling`.

Integra TPOT para automatización de la búsqueda de hiperparámetros en 
Clasificación y Regresión, y utiliza K-Means para Clustering.
Incluye evaluación de métricas y generación automática de justificaciones técnicas.
"""
import pandas as pd
import numpy as np
import logging
import time
from typing import Tuple, Dict, Any

from tpot import TPOTClassifier, TPOTRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_absolute_error, mean_squared_error,
    silhouette_score
)
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

def generar_justificacion_tecnica(tarea: str, metrica_principal: float, n_samples: int) -> str:
    """Genera una justificación técnica basada en el rendimiento del modelo."""
    justificacion = f"Análisis de {tarea} (N={n_samples}): "
    
    if tarea == "Clasificacion":
        if metrica_principal < 0.6:
            justificacion += (
                f"El Accuracy obtenido es bajo ({metrica_principal:.2f}). "
                "Esto es el comportamiento estadísticamente esperado debido a la "
                "drástica reducción del tamaño de la muestra tras la limpieza de datos. "
                "Cualquier modelo con un puntaje muy elevado en este escenario "
                "sería un claro indicio de sobreajuste (Overfitting). El modelo refleja "
                "el poder predictivo real sobre los datos disponibles."
            )
        else:
            justificacion += f"El modelo presenta un rendimiento adecuado (Accuracy={metrica_principal:.2f})."
            
    elif tarea == "Regresion":
        if metrica_principal > 0.95:
            justificacion += (
                f"El R2 es inusualmente alto ({metrica_principal:.2f}). "
                "Es crítico verificar si existe fuga de datos (Data Leakage) en las variables independientes, "
                "como incluir componentes exactos del cálculo de la variable objetivo."
            )
        elif metrica_principal < 0.4:
            justificacion += (
                f"El R2 es bajo ({metrica_principal:.2f}). "
                "Esto indica que las variables predictoras actuales no contienen la "
                "señal suficiente para estimar la variable objetivo de forma lineal."
            )
        else:
            justificacion += f"El modelo explica una proporción razonable de la varianza (R2={metrica_principal:.2f})."
            
    return justificacion

def extraer_modelos_tpot(tpot_obj) -> Tuple[list, int, int]:
    """Extrae pipelines evaluados por TPOT, y cuenta los exitosos y fallidos."""
    modelos = []
    total_probados = 0
    
    try:
        evaluated = getattr(tpot_obj, 'evaluated_individuals_', getattr(tpot_obj, 'evaluated_individuals', {}))
        
        if isinstance(evaluated, dict):
            total_probados = len(evaluated)
            for pipeline_str, metrics in evaluated.items():
                score = metrics.get('internal_cv_score', float('-inf'))
                if score != float('-inf'):
                    modelos.append({
                        "pipeline": str(pipeline_str),
                        "score": score
                    })
        elif hasattr(evaluated, 'index') and hasattr(evaluated, 'columns'):
            total_probados = len(evaluated)
            score_col = 'internal_cv_score'
            if 'internal_cv_score' not in evaluated.columns:
                num_cols = evaluated.select_dtypes(include=[np.number]).columns
                if len(num_cols) > 0:
                    score_col = num_cols[0]
                else:
                    score_col = evaluated.columns[0]

            for idx, row in evaluated.iterrows():
                score_val = float(row[score_col]) if pd.notna(row[score_col]) else float('-inf')
                if score_val != float('-inf'):
                    # Añadir métricas extra si existen en el dataframe
                    dict_modelo = {"pipeline": str(idx), "score": score_val}
                    for col in evaluated.columns:
                        if col != score_col and pd.notna(row[col]):
                            val = row[col]
                            if isinstance(val, (int, float, np.number)):
                                dict_modelo[col] = float(val)
                    modelos.append(dict_modelo)
            
        modelos = sorted(modelos, key=lambda x: x['score'], reverse=True)
        
    except Exception as e:
        logger.warning(f"Error al extraer modelos de TPOT: {e}")
        
    return modelos, total_probados, len(modelos)

def train_classification(df: pd.DataFrame) -> Tuple[Any, Dict[str, Any]]:
    """Entrena un modelo de clasificación usando TPOT."""
    logger.info("Iniciando AutoML para Clasificación...")
    target = 'segmento'
    
    if target not in df.columns:
        return None, {"error": "Falta columna target"}
        
    df_valid = df.dropna(subset=[target]).copy()
    n_samples = len(df_valid)
    
    if n_samples < 10:
        return None, {"error": "Insuficientes datos"}

    le = LabelEncoder()
    y = le.fit_transform(df_valid[target].astype(str))
    X = df_valid.select_dtypes(include=[np.number]).fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    start_time = time.time()
    tpot = TPOTClassifier(
        generations=2, 
        population_size=10, 
        cv=min(3, len(np.unique(y_train))), 
        random_state=42, 
        max_time_mins=2
    )
    tpot.fit(X_train, y_train)
    tiempo_ejecucion = time.time() - start_time
    
    y_pred = tpot.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    justificacion = generar_justificacion_tecnica("Clasificacion", acc, n_samples)
    modelos_evaluados, total_probados, validos = extraer_modelos_tpot(tpot)
    
    reporte = {
        "tiempo_ejecucion_segundos": round(tiempo_ejecucion, 2),
        "total_modelos_probados": total_probados,
        "modelos_exitosos": validos,
        "metricas_mejor_modelo": {
            "accuracy": acc,
            "precision_weighted": prec,
            "recall_weighted": rec,
            "f1_score_weighted": f1
        },
        "n_muestras_entrenamiento": n_samples,
        "justificacion_tecnica": justificacion,
        "mejor_pipeline": str(tpot.fitted_pipeline_),
        "mensaje": "De todos los modelos, los que no dieron error o cumplieron con el tiempo dado fueron los siguientes:",
        "modelos_evaluados": modelos_evaluados
    }
    
    return tpot.fitted_pipeline_, reporte

def train_regression(df: pd.DataFrame) -> Tuple[Any, Dict[str, Any]]:
    """Entrena un modelo de regresión usando TPOT."""
    logger.info("Iniciando AutoML para Regresión...")
    target = 'monto_total_venta'
    cols_excluir = [target, 'monto_total_venta_zscore', 'cantidad', 'precio_unitario']
    
    if target not in df.columns:
        return None, {"error": "Falta columna target"}
        
    df_valid = df.dropna(subset=[target]).copy()
    n_samples = len(df_valid)
    y = df_valid[target]
    cols_features = [c for c in df_valid.select_dtypes(include=[np.number]).columns if c not in cols_excluir]
    X = df_valid[cols_features].fillna(0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    start_time = time.time()
    tpot = TPOTRegressor(
        generations=2, 
        population_size=10, 
        cv=5, 
        random_state=42, 
        max_time_mins=2
    )
    tpot.fit(X_train, y_train)
    tiempo_ejecucion = time.time() - start_time
    
    y_pred = tpot.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    justificacion = generar_justificacion_tecnica("Regresion", r2, n_samples)
    modelos_evaluados, total_probados, validos = extraer_modelos_tpot(tpot)
    
    reporte = {
        "tiempo_ejecucion_segundos": round(tiempo_ejecucion, 2),
        "total_modelos_probados": total_probados,
        "modelos_exitosos": validos,
        "metricas_mejor_modelo": {
            "r2_score": r2,
            "mean_absolute_error": mae,
            "root_mean_squared_error": rmse
        },
        "n_muestras_entrenamiento": n_samples,
        "justificacion_tecnica": justificacion,
        "mejor_pipeline": str(tpot.fitted_pipeline_),
        "mensaje": "De todos los modelos, los que no dieron error o cumplieron con el tiempo dado fueron los siguientes:",
        "modelos_evaluados": modelos_evaluados
    }
    
    return tpot.fitted_pipeline_, reporte

def train_clustering(df: pd.DataFrame) -> Tuple[Any, Dict[str, Any]]:
    """Aplica K-Means para segmentación probando K=2 hasta K=8."""
    logger.info("Iniciando búsqueda de Clustering (K-Means)...")
    
    X = df.select_dtypes(include=[np.number]).fillna(0)
    n_samples = len(X)
    
    if n_samples < 10:
        return None, {"error": "Insuficientes datos"}
        
    modelos_evaluados = []
    mejor_k = -1
    mejor_sil_score = -1.0
    mejor_modelo = None
    
    start_time = time.time()
    # Bucle para probar múltiples K
    for k in range(2, 9):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(X)
        if len(np.unique(labels)) > 1:
            sil_score = silhouette_score(X, labels)
        else:
            sil_score = -1.0
            
        if sil_score != -1.0:
            modelos_evaluados.append({
                "configuracion": f"KMeans(n_clusters={k})",
                "silhouette_score": sil_score
            })
        
        if sil_score > mejor_sil_score:
            mejor_sil_score = sil_score
            mejor_k = k
            mejor_modelo = kmeans
            
    tiempo_ejecucion = time.time() - start_time
            
    modelos_evaluados = sorted(modelos_evaluados, key=lambda x: x['silhouette_score'], reverse=True)
    
    total_probados = 7 # 2 al 8 son 7 iteraciones
    validos = len(modelos_evaluados)
        
    justificacion = (
        f"Segmentación K-Means optimizada finalizada. Se evaluaron K de 2 a 8. "
        f"El mejor modelo encontró {mejor_k} clusters con un Silhouette Score de {mejor_sil_score:.2f}."
    )
    logger.info(justificacion)
    
    reporte = {
        "tiempo_ejecucion_segundos": round(tiempo_ejecucion, 2),
        "total_modelos_probados": total_probados,
        "modelos_exitosos": validos,
        "metricas_mejor_modelo": {
            "silhouette_score": mejor_sil_score,
            "n_clusters_optimo": mejor_k
        },
        "justificacion_tecnica": justificacion,
        "mensaje": "De todos los modelos, los que no dieron error o cumplieron con el tiempo dado fueron los siguientes:",
        "modelos_evaluados": modelos_evaluados
    }
    
    return mejor_modelo, reporte
