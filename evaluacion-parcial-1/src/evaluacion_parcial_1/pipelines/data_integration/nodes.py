import pandas as pd

def integrar_datos(ventas: pd.DataFrame, productos: pd.DataFrame, clientes: pd.DataFrame, devoluciones: pd.DataFrame) -> pd.DataFrame:
    # 1. Joins/merges entre las 4 tablas
    df_merged = ventas.merge(clientes, on='id_cliente', how='left')
    df_merged = df_merged.merge(productos, on='id_producto', how='left')
    df_merged = df_merged.merge(devoluciones, on='id_venta', how='left')
    
    # Rellenar nulos lógicos post-join (ventas que no tuvieron devolución)
    if 'estado' in df_merged.columns:
        df_merged['estado'] = df_merged['estado'].fillna('Sin Devolucion')
    if 'motivo' in df_merged.columns:
        df_merged['motivo'] = df_merged['motivo'].fillna('Ninguno')

    # 2. Creación de features derivadas
    if 'precio_unitario' in df_merged.columns and 'cantidad' in df_merged.columns:
        df_merged['monto_total_venta'] = df_merged['precio_unitario'] * df_merged['cantidad']

    # 3. Transformaciones avanzadas (groupby)
    # Creamos una columna que calcule el total histórico gastado por cada cliente
    if 'id_cliente' in df_merged.columns and 'monto_total_venta' in df_merged.columns:
        gasto_historico = df_merged.groupby('id_cliente')['monto_total_venta'].transform('sum')
        df_merged['gasto_historico_cliente'] = gasto_historico

    # 4. Codificación de variables categóricas (One-Hot Encoding)
    # Convertimos 'canal_venta' en columnas binarias (1 o 0)
    if 'canal_venta' in df_merged.columns:
        dummies = pd.get_dummies(df_merged['canal_venta'], prefix='canal')
        dummies = dummies.astype(int) # Asegurar que sean 1 y 0, no True/False
        df_merged = pd.concat([df_merged, dummies], axis=1)
        df_merged = df_merged.drop('canal_venta', axis=1)

    # 5. Normalización/estandarización de columnas numéricas (Z-score)
    cols_to_scale = ['precio_unitario', 'monto_total_venta']
    for col in cols_to_scale:
        if col in df_merged.columns:
            mean = df_merged[col].mean()
            std = df_merged[col].std()
            if std > 0:
                df_merged[f'{col}_zscore'] = (df_merged[col] - mean) / std
            else:
                df_merged[f'{col}_zscore'] = 0

    return df_merged