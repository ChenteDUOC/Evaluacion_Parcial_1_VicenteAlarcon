import pandas as pd
import numpy as np

def remove_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Elimina outliers de una columna numérica usando el método IQR."""
    if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Filtramos el dataframe conservando solo los valores dentro de los límites
        df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        return df_filtered
    return df

def preprocess_ventas(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy().drop_duplicates()
    df_clean = df_clean.dropna(subset=['id_venta', 'id_cliente', 'id_producto'])
    
    df_clean['fecha'] = pd.to_datetime(df_clean['fecha'], dayfirst=True, errors='coerce')
    df_clean = df_clean.dropna(subset=['fecha'])
    
    cols_str = ['metodo_pago', 'canal_venta']
    for col in cols_str:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip().str.title()
            df_clean[col] = df_clean[col].replace('Nan', np.nan)
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
            
    cols_num = ['precio_unitario', 'cantidad']
    for col in cols_num:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            
    # Tratamiento de outliers (IQR)
    df_clean = remove_outliers_iqr(df_clean, 'precio_unitario')
    df_clean = remove_outliers_iqr(df_clean, 'cantidad')
    
    return df_clean

def preprocess_productos(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy().drop_duplicates()
    df_clean = df_clean.dropna(subset=['id_producto']) 
    
    cols_str = ['categoria', 'subcategoria', 'proveedor']
    for col in cols_str:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip().str.title()
            df_clean[col] = df_clean[col].replace('Nan', np.nan)
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
            
    cols_num = ['precio_lista', 'stock']
    for col in cols_num:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            
    # Tratamiento de outliers (IQR)
    df_clean = remove_outliers_iqr(df_clean, 'precio_lista')
    df_clean = remove_outliers_iqr(df_clean, 'stock')
            
    return df_clean

def preprocess_clientes(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy().drop_duplicates()
    df_clean = df_clean.dropna(subset=['id_cliente'])
    
    df_clean['fecha_registro'] = pd.to_datetime(df_clean['fecha_registro'], dayfirst=True, errors='coerce')
    df_clean = df_clean.dropna(subset=['fecha_registro'])
    
    cols_str = ['region', 'ciudad', 'segmento']
    for col in cols_str:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip().str.title()
            df_clean[col] = df_clean[col].replace('Nan', np.nan)
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
            
    return df_clean

def preprocess_devoluciones(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy().drop_duplicates()
    df_clean = df_clean.dropna(subset=['id_devolucion', 'id_venta'])
    
    df_clean['fecha_devolucion'] = pd.to_datetime(df_clean['fecha_devolucion'], dayfirst=True, errors='coerce')
    df_clean = df_clean.dropna(subset=['fecha_devolucion'])
    
    cols_str = ['motivo', 'estado']
    for col in cols_str:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip().str.title()
            df_clean[col] = df_clean[col].replace('Nan', np.nan)
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
            
    return df_clean