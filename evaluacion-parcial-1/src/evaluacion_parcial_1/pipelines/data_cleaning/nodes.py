"""
Módulo de limpieza de datos para el pipeline `data_cleaning`.

Aplica técnicas avanzadas de imputación de nulos y manejo de valores atípicos
(Winsorización) para asegurar la integridad de los datos sin perder información.
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def winsorize_column(df: pd.DataFrame, column: str, lower_quantile: float = 0.05, upper_quantile: float = 0.95) -> pd.DataFrame:
    """Aplica Winzorización a una columna numérica para suavizar outliers sin eliminar filas.
    
    Reemplaza los valores extremos por los límites del percentil definido.
    
    Args:
        df: DataFrame de entrada.
        column: Nombre de la columna a Winzorizar.
        lower_quantile: Percentil inferior de corte.
        upper_quantile: Percentil superior de corte.
        
    Returns:
        DataFrame con la columna Winzorizada.
    """
    if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
        lower_bound = df[column].quantile(lower_quantile)
        upper_bound = df[column].quantile(upper_quantile)
        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
        logger.info(f"Winzorización aplicada en {column}: [{lower_bound}, {upper_bound}]")
    return df

def smart_impute(series: pd.Series) -> pd.Series:
    """Imputa nulos estratégicamente según el tipo de datos y distribución."""
    if not series.isnull().any():
        return series
    
    if pd.api.types.is_numeric_dtype(series):
        # Usamos mediana si hay asimetría (skew), si no, la media.
        skewness = series.skew()
        if abs(skewness) > 1.0:
            return series.fillna(series.median())
        return series.fillna(series.mean())
    else:
        # Imputación por moda para categóricas
        mode_val = series.mode()
        return series.fillna(mode_val[0] if not mode_val.empty else "Desconocido")

def preprocess_ventas(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza de la tabla ventas."""
    df_clean = df.copy().drop_duplicates()
    df_clean = df_clean.dropna(subset=['id_venta', 'id_cliente', 'id_producto'])
    
    df_clean['fecha'] = pd.to_datetime(df_clean['fecha'], dayfirst=True, errors='coerce')
    df_clean = df_clean.dropna(subset=['fecha'])
    
    cols_str = ['metodo_pago', 'canal_venta']
    for col in cols_str:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip().str.title().replace('Nan', np.nan)
            df_clean[col] = smart_impute(df_clean[col])
            
    cols_num = ['precio_unitario', 'cantidad']
    for col in cols_num:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            df_clean[col] = smart_impute(df_clean[col])
            
    df_clean = winsorize_column(df_clean, 'precio_unitario')
    df_clean = winsorize_column(df_clean, 'cantidad')
    return df_clean

def preprocess_productos(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza de la tabla productos."""
    df_clean = df.copy().drop_duplicates()
    df_clean = df_clean.dropna(subset=['id_producto']) 
    
    cols_str = ['categoria', 'subcategoria', 'proveedor']
    for col in cols_str:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip().str.title().replace('Nan', np.nan)
            df_clean[col] = smart_impute(df_clean[col])
            
    cols_num = ['precio_lista', 'stock']
    for col in cols_num:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            df_clean[col] = smart_impute(df_clean[col])
            
    df_clean = winsorize_column(df_clean, 'precio_lista')
    df_clean = winsorize_column(df_clean, 'stock')
    return df_clean

def preprocess_clientes(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza de la tabla clientes."""
    df_clean = df.copy().drop_duplicates()
    df_clean = df_clean.dropna(subset=['id_cliente'])
    
    df_clean['fecha_registro'] = pd.to_datetime(df_clean['fecha_registro'], dayfirst=True, errors='coerce')
    df_clean = df_clean.dropna(subset=['fecha_registro'])
    
    cols_str = ['region', 'ciudad', 'segmento']
    for col in cols_str:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip().str.title().replace('Nan', np.nan)
            df_clean[col] = smart_impute(df_clean[col])
    return df_clean

def preprocess_devoluciones(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza de la tabla devoluciones."""
    df_clean = df.copy().drop_duplicates()
    df_clean = df_clean.dropna(subset=['id_devolucion', 'id_venta'])
    
    df_clean['fecha_devolucion'] = pd.to_datetime(df_clean['fecha_devolucion'], dayfirst=True, errors='coerce')
    df_clean = df_clean.dropna(subset=['fecha_devolucion'])
    
    cols_str = ['motivo', 'estado']
    for col in cols_str:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip().str.title().replace('Nan', np.nan)
            df_clean[col] = smart_impute(df_clean[col])
    return df_clean