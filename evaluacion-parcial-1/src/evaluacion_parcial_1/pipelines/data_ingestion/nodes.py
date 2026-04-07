import pandas as pd
from typing import Dict, Any

def generar_diagnostico(ventas: pd.DataFrame, productos: pd.DataFrame, clientes: pd.DataFrame, devoluciones: pd.DataFrame) -> Dict[str, Any]:
    datasets = {"ventas": ventas, "productos": productos, "clientes": clientes, "devoluciones": devoluciones}
    reporte = {}
    
    for nombre, df in datasets.items():
        # Exploración básica y detección de problemas
        reporte[nombre] = {
            "filas_columnas": df.shape,
            "nulos_totales": int(df.isnull().sum().sum()),
            "tipos_de_datos": df.dtypes.astype(str).to_dict(),
            "columnas": df.columns.tolist()
        }
        
    return reporte