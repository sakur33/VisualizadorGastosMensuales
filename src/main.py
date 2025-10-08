# /src/main.py

import calendar
from fnmatch import fnmatch
import shutil
import argparse
import os
import glob as _glob
from pathlib import Path
from typing import List
import pandas as pd
import visualizar as visualizar
import datos as datos
import utils as utils


def ejecutar_por_mes(df_all: pd.DataFrame, out_root: Path):
    for _, df_mes in df_all.groupby(df_all["Fecha"].dt.to_period("M")):
        if df_mes.empty:
            continue

        # Rango real de datos en ese mes
        f_ini_dt = pd.to_datetime(df_mes["Fecha"].min()).normalize()
        f_fin_dt = pd.to_datetime(df_mes["Fecha"].max()).normalize()
        str_fechas = (
            f"{f_ini_dt.strftime('%Y-%m-%d')} - {f_fin_dt.strftime('%Y-%m-%d')}"
        )

        # ¿Mes completo? (día 1 y último día del mes)
        last_day = calendar.monthrange(f_ini_dt.year, f_ini_dt.month)[1]
        es_completo = (f_ini_dt.day == 1) and (f_fin_dt.day == last_day)

        # Nombre de carpeta
        if es_completo:
            carpeta_mes = f_ini_dt.strftime("%Y-%m")
        else:
            carpeta_mes = (
                f"{f_ini_dt.strftime('%Y-%m-%d')}_a_{f_fin_dt.strftime('%Y-%m-%d')}"
            )

        out_dir_mes = out_root / carpeta_mes
        out_dir_mes.mkdir(parents=True, exist_ok=True)

        # Vistas mensuales
        visualizar.vista_resumen_mes(df_mes, out_dir_mes, str_fechas)
        visualizar.vista_flujo_diario(df_mes, out_dir_mes, str_fechas)
        visualizar.vista_por_tipo(df_mes, out_dir_mes, str_fechas)
        visualizar.vista_por_tipo_general(df_mes, out_dir_mes, str_fechas)
        visualizar.vista_top_gastos(
            df_mes, out_dir_mes, top_n=10, str_fechas=str_fechas
        )
        visualizar.vista_por_semana(df_mes, out_dir_mes, str_fechas)
        visualizar.vista_media_por_dia_semana(df_mes, out_dir_mes, str_fechas)
        visualizar.vista_barras_por_tipo_detalle(df_mes, out_dir_mes, str_fechas)
        visualizar.vista_histograma_gastos(df_mes, out_dir_mes, str_fechas)

        datos.exportar_tablas_utiles(df_mes, out_dir_mes, str_fechas=str_fechas)


def ejecutar_intermes(df_all: pd.DataFrame, out_root: Path, str_fechas: str):
    inter_dir = out_root / "00_intermes"
    inter_dir.mkdir(parents=True, exist_ok=True)
    visualizar.vista_totales_por_mes(df_all, inter_dir, str_fechas)
    visualizar.vista_mes_por_tipo_general(df_all, inter_dir, str_fechas)
    # visualizar.vista_heatmap_dia_semana_mes(df_all, inter_dir, str_fechas)
    # visualizar.vista_boxplot_importes_por_mes(df_all, inter_dir, str_fechas)


def main():

    data_path = "./data/*.xlsx"
    config_path = "./config/categorias_de_gasto.json"
    out_root = "./salidas"
    paths: List[str] = []
    try:
        paths.extend(_glob.glob(str(utils.resolve_path(data_path))))
    except Exception as e:
        print(e)

    # Construir lista de EXCLUSIONES
    ex_patterns = ["*/test*.xlsx", "*/~$*.xlsx"]

    def _is_excluded(p: str, patterns: List[str]) -> bool:
        np = os.path.normpath(p)
        return any(fnmatch(np, pat) for pat in patterns)

    # Filtrar rutas excluidas
    paths = [p for p in paths if not _is_excluded(p, ex_patterns)]

    if not paths:
        raise SystemExit("Tras aplicar exclusiones no quedan ficheros para procesar.")

    # Cargar y filtrar
    df_all = datos.cargar_multiples_xls(paths, config_path=config_path)
    df_all = df_all.drop_duplicates()
    datos.describir_no_categorizados(df_all, config_path)
    if df_all.empty:
        raise SystemExit("No hay datos tras el filtrado.")

    # Título/raíz de salida por rango real de datos
    f_ini = df_all["Fecha"].min().strftime("%Y-%m-%d")
    f_fin = df_all["Fecha"].max().strftime("%Y-%m-%d")
    str_fechas = f"{f_ini} - {f_fin}"

    out_root = utils.resolve_path(out_root)
    utils.asegurar_dir(out_root)

    # Vistas intermes y por mes
    ejecutar_intermes(df_all, out_root, str_fechas)
    # ejecutar_por_mes(df_all, out_root)

    print("\nListo. Revisa la carpeta de salidas:")
    print(f" - Intermes: {out_root / '00_intermes'}")
    print(f" - Meses:    {out_root}")


if __name__ == "__main__":
    main()
