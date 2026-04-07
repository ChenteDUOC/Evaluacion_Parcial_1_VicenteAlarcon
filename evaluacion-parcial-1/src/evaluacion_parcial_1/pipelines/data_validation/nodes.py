import pandas as pd
from typing import Dict, Any

def validar_calidad_datos(df_final: pd.DataFrame, df_raw: pd.DataFrame) -> Dict[str, Any]:
    # 1. Comparación antes/después (Tomamos ventas como base)
    filas_antes = len(df_raw)
    filas_despues = len(df_final)
    
    # 2. Verificación de integridad post-transformación
    nulos_restantes = int(df_final.isnull().sum().sum())
    duplicados = int(df_final.duplicated().sum())
    
    # 3. Validación de esquemas (verificamos que las columnas creadas existan y los tipos de datos)
    columnas_esperadas = ['monto_total_venta', 'gasto_historico_cliente', 'precio_unitario_zscore']
    columnas_actuales = df_final.columns.tolist()
    esquema_valido = all(col in columnas_actuales for col in columnas_esperadas)
    
    reporte = {
        "comparacion_antes_despues": {
            "filas_iniciales_ventas": filas_antes,
            "filas_finales_integradas": filas_despues,
            "variacion_filas": filas_despues - filas_antes
        },
        "integridad": {
            "nulos_totales_restantes": nulos_restantes,
            "duplicados_totales": duplicados
        },
        "esquema": {
            "validacion_columnas_nuevas": "Exitosa" if esquema_valido else "Fallida",
            "columnas_totales": len(columnas_actuales),
            "tipos_de_datos": df_final.dtypes.astype(str).to_dict()
        },
        "estado_general": "Aprobado" if esquema_valido else "Con Observaciones"
    }
    
    return reporte