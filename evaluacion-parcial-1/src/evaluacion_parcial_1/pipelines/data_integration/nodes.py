"""
Módulo de integración de datos para el pipeline `data_integration`.

Realiza los cruces de tablas, ingeniería de atributos, codificación de variables
categóricas, eliminación de columnas de ruido y escalamiento de atributos numéricos.
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

def integrar_datos(ventas: pd.DataFrame, productos: pd.DataFrame, clientes: pd.DataFrame, devoluciones: pd.DataFrame) -> pd.DataFrame:
    """Unifica los datasets y prepara el feature set para modelado.
    
    Args:
        ventas: Dataset limpio de ventas.
        productos: Dataset limpio de productos.
        clientes: Dataset limpio de clientes.
        devoluciones: Dataset limpio de devoluciones.
        
    Returns:
        DataFrame integrado, escalado y listo para ser usado por los modelos de ML.
    """
    # 1. Joins/merges
    df_merged = ventas.merge(clientes, on='id_cliente', how='left')
    df_merged = df_merged.merge(productos, on='id_producto', how='left')
    df_merged = df_merged.merge(devoluciones, on='id_venta', how='left')
    
    # 2. Imputación post-join
    if 'estado' in df_merged.columns:
        df_merged['estado'] = df_merged['estado'].fillna('Sin Devolucion')
    if 'motivo' in df_merged.columns:
        df_merged['motivo'] = df_merged['motivo'].fillna('Ninguno')

    # 3. Ingeniería de atributos (Feature Engineering)
    if 'precio_unitario' in df_merged.columns and 'cantidad' in df_merged.columns:
        df_merged['monto_total_venta'] = df_merged['precio_unitario'] * df_merged['cantidad']

    if 'id_cliente' in df_merged.columns and 'monto_total_venta' in df_merged.columns:
        gasto_historico = df_merged.groupby('id_cliente')['monto_total_venta'].transform('sum')
        df_merged['gasto_historico_cliente'] = gasto_historico

    # 4. Codificación One-Hot
    if 'canal_venta' in df_merged.columns:
        dummies = pd.get_dummies(df_merged['canal_venta'], prefix='canal')
        dummies = dummies.astype(int)
        df_merged = pd.concat([df_merged, dummies], axis=1)

    # 5. Selección de atributos (Eliminación de ruido e IDs)
    cols_to_drop = [
        'id_venta', 'id_cliente', 'id_producto', 'id_devolucion', 
        'fecha', 'fecha_registro', 'fecha_devolucion', 
        'nombre', 'email', 'canal_venta'
    ]
    cols_to_drop = [c for c in cols_to_drop if c in df_merged.columns]
    df_merged = df_merged.drop(columns=cols_to_drop, errors='ignore')
    
    # Eliminar posibles columnas extra de strings libres que no aportan valor
    # Solo dejaremos variables numéricas y el target 'segmento' si existe.
    
    # 6. Escalamiento (StandardScaler) a las variables numéricas
    cols_numericas = df_merged.select_dtypes(include=['float64', 'int64', 'int32']).columns
    # Excluimos 'segmento' de ser numérico si se transformó o cualquier flag binaria pura
    scaler = StandardScaler()
    
    cols_to_scale = [c for c in cols_numericas if c not in ['segmento'] and not c.startswith('canal_')]
    if len(cols_to_scale) > 0:
        df_merged[cols_to_scale] = scaler.fit_transform(df_merged[cols_to_scale])
        logger.info(f"StandardScaler aplicado a las columnas: {cols_to_scale}")

    return df_merged