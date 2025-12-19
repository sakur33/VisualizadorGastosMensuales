import calendar
import glob as _glob
from pathlib import Path
from typing import List, Optional
import argparse

import pandas as pd

import visualizar as visualizar
import datos as datos
import utils as utils


def ejecutar_por_mes(df_all: pd.DataFrame, out_root: Path):
    for _, df_mes in df_all.groupby(df_all["Fecha"].dt.to_period("M")):
        if df_mes.empty:
            continue

        f_ini_dt = pd.to_datetime(df_mes["Fecha"].min()).normalize()
        f_fin_dt = pd.to_datetime(df_mes["Fecha"].max()).normalize()
        str_fechas = f"{f_ini_dt.strftime('%Y_%m_%d')} a {f_fin_dt.strftime('%Y_%m_%d')}"

        last_day = calendar.monthrange(f_ini_dt.year, f_ini_dt.month)[1]
        es_completo = (f_ini_dt.day == 1) and (f_fin_dt.day == last_day)

        if es_completo:
            carpeta_mes = f_ini_dt.strftime("%Y_%m")
        else:
            carpeta_mes = f"{f_ini_dt.strftime('%Y_%m_%d')}_a_{f_fin_dt.strftime('%Y_%m_%d')}"

        out_dir_mes = out_root / carpeta_mes
        out_dir_mes.mkdir(parents=True, exist_ok=True)

        visualizar.vista_resumen_mes(df_mes, out_dir_mes, str_fechas)
        visualizar.vista_flujo_diario(df_mes, out_dir_mes, str_fechas)
        visualizar.vista_por_tipo(df_mes, out_dir_mes, str_fechas)
        visualizar.vista_por_tipo_general(df_mes, out_dir_mes, str_fechas)
        visualizar.vista_top_gastos(df_mes, out_dir_mes, top_n=10, str_fechas=str_fechas)
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


def _parse_args():
    p = argparse.ArgumentParser(prefix_chars="/")

    p.add_argument("/glob", dest="glob_pattern", default="./data/mensuales/*.pdf")
    p.add_argument("/config", dest="config_path", default="./config/categorias_de_gasto.json")
    p.add_argument("/salidas", dest="salidas", default="./salidas")
    p.add_argument("/por_mes", dest="por_mes", action="store_true")

    return p.parse_args()


def main():
    args = _parse_args()

    config_path = args.config_path
    out_root = utils.resolve_path(args.salidas)
    utils.asegurar_dir(out_root)

    paths: List[str] = []
    try:
        paths.extend(_glob.glob(str(utils.resolve_path(args.glob_pattern))))
    except Exception as e:
        print(e)

    if not paths:
        raise SystemExit("No hay ficheros para procesar")

    df_all = datos.cargar_recibos_mensuales(paths, config_path=config_path)
    df_all = df_all.drop_duplicates()

    datos.describir_no_categorizados(df_all, config_path)
    datos.exportar_no_categorizados(df_all, out_root / "no_categorizados", top_n=80)

    if df_all.empty:
        raise SystemExit("No hay datos tras el filtrado")

    f_ini = df_all["Fecha"].min().strftime("%Y_%m_%d")
    f_fin = df_all["Fecha"].max().strftime("%Y_%m_%d")
    str_fechas = f"{f_ini} a {f_fin}"

    ejecutar_intermes(df_all, out_root, str_fechas)

    if args.por_mes:
        ejecutar_por_mes(df_all, out_root)

    print("\nListo Revisa la carpeta de salidas")
    print(f"Intermes {out_root / '00_intermes'}")
    print(f"Raiz {out_root}")


if __name__ == "__main__":
    main()
