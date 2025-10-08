# /src/datos.py

from datetime import datetime
import shutil
import json
import unicodedata
import re
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import utils as utils


def cargar_multiples_xls(paths: List[str], config_path: str) -> pd.DataFrame:
    # Cargamos reglas una sola vez para consistencia entre ficheros
    with open(utils.resolve_path(config_path), "r", encoding="utf-8") as f:
        raw_rules = json.load(f)
    rules = _flatten_rules(raw_rules)

    dfs = []
    for p in paths:
        dx = leer_y_normalizar(p, config_path=config_path, rules=rules)
        dfs.append(dx)
    if not dfs:
        raise ValueError("No se cargaron ficheros.")
    df_all = pd.concat(dfs, ignore_index=True).sort_values("Fecha")

    # Añadimos periodos mensual
    df_all["AñoMes"] = df_all["Fecha"].dt.to_period("M")
    df_all["AñoMesStr"] = df_all["AñoMes"].astype(str)  # "YYYY-MM"
    return df_all


def filtrar_por_rango(
    df: pd.DataFrame, inicio: Optional[str], fin: Optional[str]
) -> pd.DataFrame:
    if inicio:
        df = df[df["Fecha"] >= pd.to_datetime(inicio)]
    if fin:
        df = df[df["Fecha"] <= pd.to_datetime(fin)]
    return df


def exportar_tablas_utiles(df: pd.DataFrame, out_dir: Path, str_fechas: str):
    df.to_csv(out_dir / f"movimientos_normalizados_{str_fechas}.csv", index=False)
    (
        df.groupby(["Fecha", "Tipo"])["Gasto"]
        .sum()
        .reset_index()
        .to_csv(out_dir / f"gasto_por_fecha_y_tipo_{str_fechas}.csv", index=False)
    )
    (
        df.groupby("Fecha")[["Gasto", "Ingreso"]]
        .sum()
        .reset_index()
        .to_csv(out_dir / f"flujo_diario_{str_fechas}.csv", index=False)
    )
    print("✅ Guardadas tablas CSV auxiliares.")


def describir_no_categorizados(df_all: pd.DataFrame, config_path):
    dfx = df_all.copy()
    dfnc = dfx[dfx["Tipo"].fillna("").str.strip().eq("Sin categorizar")].copy()
    if dfnc.empty:
        print("✅ No hay movimientos sin categorizar. No se modifica el JSON.")
        return

    pares = (
        dfnc[["Concepto", "Movimiento"]]
        .drop_duplicates()
        .fillna({"Concepto": "", "Movimiento": ""})
    )

    cfg_path = Path(config_path)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)

    # Carga JSON
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        print("ℹ️ No existe el JSON de categorías. Se creará uno nuevo.")
        cfg = []

    # --- Helpers de ordenación ---
    def _sort_casefold(xs):
        return sorted(xs, key=lambda s: _norm(s))

    def _ordenar_cfg_grouped(cfg_list):
        # Ordena grupos por General, Tipos por Tipo y listas de patrones
        for g in cfg_list:
            tipos = g.get("Tipos", [])
            for tb in tipos:
                pats = tb.get("pattern_substring", [])
                if isinstance(pats, str):
                    pats = [pats]
                tb["pattern_substring"] = _sort_casefold(list(dict.fromkeys(pats)))
            g["Tipos"] = sorted(tipos, key=lambda tb: _norm(tb.get("Tipo", "")))
        cfg_list.sort(key=lambda g: _norm(g.get("General", "")))
        return cfg_list

    # --- Construye set de patrones existentes normalizados ---
    existentes = set()
    if _is_grouped_config(cfg) or (isinstance(cfg, dict) and _is_grouped_config(cfg)):
        cfg_list = cfg if isinstance(cfg, list) else [cfg]
        for g in cfg_list:
            for tb in g.get("Tipos", []):
                pats = tb.get("pattern_substring", [])
                if isinstance(pats, str):
                    pats = [pats]
                for p in pats:
                    existentes.add(_norm(p))
    else:
        # Formato legado
        if not isinstance(cfg, list):
            cfg = []
        for item in cfg:
            if isinstance(item, dict):
                existentes.add(_norm(item.get("pattern_substring", "")))

    # --- Candidatos nuevos ---
    nuevos_raw = []
    for _, row in pares.iterrows():
        concepto = str(row["Concepto"] or "").strip()
        movimiento = str(row["Movimiento"] or "").strip()
        raw_pat = (concepto + " " + movimiento).strip()
        if not raw_pat:
            continue
        pat_norm = _norm(raw_pat)
        if pat_norm and pat_norm not in existentes:
            nuevos_raw.append(raw_pat)
            existentes.add(pat_norm)

    if not nuevos_raw:
        print("ℹ️ No hay patrones nuevos que añadir.")
        # Aun así, si es agrupado, lo reordenamos de forma bonita
        if _is_grouped_config(cfg) or (
            isinstance(cfg, dict) and _is_grouped_config(cfg)
        ):
            cfg_list = cfg if isinstance(cfg, list) else [cfg]
            cfg_list = _ordenar_cfg_grouped(cfg_list)
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg_list, f, ensure_ascii=False, indent=4)
            print(f"✅ Reordenado {cfg_path} (General/Tipo/patrones).")
        else:
            # legado: reordenar por Tipo y patrón
            if isinstance(cfg, list) and cfg:
                cfg_ordenado = sorted(
                    cfg,
                    key=lambda it: (
                        _norm(it.get("Tipo", "")),
                        _norm(it.get("pattern_substring", "")),
                    ),
                )
                with open(cfg_path, "w", encoding="utf-8") as f:
                    json.dump(cfg_ordenado, f, ensure_ascii=False, indent=4)
                print(f"✅ Reordenado {cfg_path} por Tipo/patrón.")
        return

    # --- Backup ---
    if cfg_path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = cfg_path.with_name(f"{cfg_path.stem}.backup_{ts}{cfg_path.suffix}")
        try:
            shutil.copy(cfg_path, backup)
            print(f"🗂️  Backup creado: {backup}")
        except Exception as e:
            print(f"⚠️ No se pudo crear backup: {e}")

    # --- Escribir en el formato adecuado ---
    if _is_grouped_config(cfg) or (isinstance(cfg, dict) and _is_grouped_config(cfg)):
        cfg_list = cfg if isinstance(cfg, list) else [cfg]

        # Asegura grupo "Sin categorizar"
        gen_key = "Sin categorizar"
        tipo_key = "Sin categorizar"

        g_idx = next(
            (
                i
                for i, g in enumerate(cfg_list)
                if _norm(g.get("General", "")) == _norm(gen_key)
            ),
            None,
        )
        if g_idx is None:
            cfg_list.append({"General": gen_key, "Tipos": []})
            g_idx = len(cfg_list) - 1

        # Asegura Tipo "Sin categorizar" dentro del grupo
        tipos = cfg_list[g_idx].setdefault("Tipos", [])
        t_idx = next(
            (
                i
                for i, tb in enumerate(tipos)
                if _norm(tb.get("Tipo", "")) == _norm(tipo_key)
            ),
            None,
        )
        if t_idx is None:
            tipos.append({"Tipo": tipo_key, "pattern_substring": []})
            t_idx = len(tipos) - 1

        # Fusiona y ordena patrones (dedupe por normalización)
        ya = tipos[t_idx].get("pattern_substring", [])
        if isinstance(ya, str):
            ya = [ya]
        # dedupe respetando el primero visto
        merged = list(ya) + list(nuevos_raw)
        # quita duplicados por _norm manteniendo orden
        seen = set()
        dedup = []
        for p in merged:
            n = _norm(p)
            if n not in seen:
                seen.add(n)
                dedup.append(p)

        tipos[t_idx]["pattern_substring"] = _sort_casefold(dedup)
        cfg_list = _ordenar_cfg_grouped(cfg_list)

        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg_list, f, ensure_ascii=False, indent=4)

        print(
            f"✅ Actualizado {cfg_path}: añadidas {len(nuevos_raw)} nuevas reglas a ‘{gen_key} / {tipo_key}’."
        )
    else:
        # Legado: lista plana
        cfg_actualizado = list(cfg)
        for raw_pat in nuevos_raw:
            cfg_actualizado.append(
                {
                    "pattern_substring": raw_pat,
                    "Tipo": "Sin categorizar",
                    "General": "Sin categorizar",
                }
            )
        cfg_ordenado = sorted(
            cfg_actualizado,
            key=lambda it: (
                _norm(it.get("Tipo", "")),
                _norm(it.get("pattern_substring", "")),
            ),
        )
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg_ordenado, f, ensure_ascii=False, indent=4)
        print(
            f"✅ Actualizado {cfg_path}: añadidas {len(nuevos_raw)} nuevas reglas (formato legado)."
        )


def parse_excel_date_series(s: pd.Series) -> pd.Series:
    """Convierte una serie Mixta (serial Excel / str DD/MM/YYYY / datetime) a datetime64[ns]."""
    if s is None:
        return pd.Series(dtype="datetime64[ns]")

    # Si ya es datetime, solo coerciona y devuelve
    if pd.api.types.is_datetime64_any_dtype(s):
        out = pd.to_datetime(s, errors="coerce")
        return out

    # Prepara salida
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    # 1) Seriales de Excel (números)
    nums = pd.to_numeric(s, errors="coerce")
    num_mask = nums.notna()
    if num_mask.any():
        out.loc[num_mask] = pd.to_datetime(
            nums.loc[num_mask],
            unit="D",
            origin="1899-12-30",  # sistema 1900 de Excel
            errors="coerce",
        )

    # 2) Strings (DD/MM/YYYY, etc.)
    str_mask = ~num_mask
    if str_mask.any():
        # Limpia strings y convierte con dayfirst=True
        st = s.loc[str_mask].astype(str).str.strip().replace({"": None})
        out.loc[str_mask] = pd.to_datetime(st, dayfirst=True, errors="coerce")

    return out


def _norm(s: str) -> str:
    s = str(s or "")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def _is_grouped_config(cfg) -> bool:
    # Formato nuevo:
    # [
    #   {"General":"...", "Tipos":[{"Tipo":"...", "pattern_substring":[...]} , ...]},
    #   ...
    # ]
    if isinstance(cfg, dict):
        return ("General" in cfg) and ("Tipos" in cfg)
    if isinstance(cfg, list) and cfg and isinstance(cfg[0], dict):
        return "Tipos" in cfg[0] and "General" in cfg[0]
    return False


def _flatten_rules(cfg) -> List[Tuple[str, str, str]]:
    """
    Devuelve lista de tuplas normalizadas (pat_norm, Tipo, General).
    Soporta:
      1) NUEVO formato agrupado por General/Tipos/pattern_substring[]
      2) LEGADO: lista de dicts con keys: pattern_substring (str), Tipo, General
      3) LEGADO: lista de tuplas (pat, tipo, general)
    """
    rules: List[Tuple[str, str, str]] = []

    # Normaliza a lista de grupos si viene como dict único
    if isinstance(cfg, dict) and _is_grouped_config(cfg):
        cfg = [cfg]

    if _is_grouped_config(cfg):
        # Nuevo formato
        for group in cfg:
            general = str(group.get("General", "")).strip() or "(Sin tipo)"
            for tipo_block in group.get("Tipos", []):
                tipo = str(tipo_block.get("Tipo", "")).strip() or "(Sin tipo)"
                pats = tipo_block.get("pattern_substring", [])
                if isinstance(pats, str):
                    pats = [pats]
                for p in pats:
                    pat = _norm(p)
                    if pat:
                        rules.append((pat, tipo, general))
        return rules

    # Formatos legados
    if isinstance(cfg, list):
        for r in cfg:
            if isinstance(r, dict):
                pat = _norm(r.get("pattern_substring", ""))
                tipo = str(r.get("Tipo", "(Sin tipo)")).strip() or "(Sin tipo)"
                general = str(r.get("General", "(Sin tipo)"))
                if pat:
                    rules.append((pat, tipo, general))
            else:
                # tupla/lista: (pat, tipo, general)
                pat = _norm(r[0])
                if pat:
                    rules.append((pat, r[1], r[2]))
    return rules


def categorizar_gastos(row: pd.Series, rules: List[Tuple[str, str, str]]):
    """
    Devuelve (tipo, tipo_general) usando el patrón MÁS LARGO que aparezca
    en el texto normalizado de Concepto + Movimiento.
    En caso de empate de longitud, se queda el primero según el orden en 'rules'.
    """
    texto = _norm(f"{row.get('Concepto', '')} {row.get('Movimiento', '')}")
    mejor = ("Sin categorizar", "Sin categorizar")
    mejor_len = -1
    matches = []
    for pat, tipo, tipo_general in rules:
        if pat and pat in texto and len(pat) > mejor_len:
            matches.append(pat)
            mejor = (tipo, tipo_general)
            mejor_len = len(pat)
    if len(matches) > 1:
        pass
    return mejor


def leer_y_normalizar(
    xls_path: str,
    config_path: str,
    rules: Optional[List[Tuple[str, str, str]]] = None,
) -> pd.DataFrame:
    df = pd.read_excel(
        xls_path,
        header=4,
        dtype={"F.Valor": object, "Fecha": object},
        engine="openpyxl",
    )
    # Carga reglas si no vienen desde fuera
    if rules is None:
        with open(utils.resolve_path(config_path), "r", encoding="utf-8") as f:
            raw_rules = json.load(f)
        rules = _flatten_rules(raw_rules)

    # Clasificación por reglas (Tipo, Tipo General)
    df[["Tipo", "Tipo General"]] = df.apply(
        categorizar_gastos,
        axis=1,
        args=(rules,),
        result_type="expand",
    )
    # Limpieza columnas y fechas
    df = df.drop(
        columns=[c for c in df.columns if "Unnamed" in str(c)], errors="ignore"
    )
    # ✅ Parseo robusto de fechas (DD/MM/YYYY o serial Excel)
    df["F.Valor"] = parse_excel_date_series(df.get("F.Valor"))
    df["Fecha"] = parse_excel_date_series(df.get("Fecha"))

    # Normaliza a medianoche
    if "F.Valor" in df.columns:
        df["F.Valor"] = df["F.Valor"].dt.normalize()
    if "Fecha" in df.columns:
        df["Fecha"] = df["Fecha"].dt.normalize()

    # Importe a numérico y drop de nulos esenciales
    df["Importe"] = pd.to_numeric(df["Importe"], errors="coerce")
    df = df.dropna(subset=["Fecha", "Importe"]).copy()

    # Limpieza strings
    for col in ["Tipo", "Categoría", "Concepto", "Tipo General"]:
        if col in df.columns:
            df[col] = df[col].fillna("").str.strip()

    # Señales
    es_ahorro = (
        df.get("Tipo General", "").eq("Ahorro")
        if "Tipo General" in df.columns
        else pd.Series(False, index=df.index)
    )
    es_gasto = (
        ~df.get("Tipo General", "").isin(["Ahorro", "Ingresos", "Desahorro"])
        if "Tipo General" in df.columns
        else pd.Series(False, index=df.index)
    )
    es_ingreso = (
        df.get("Tipo General", "").eq("Ingresos")
        if "Tipo General" in df.columns
        else pd.Series(False, index=df.index)
    )
    es_desahorro = (
        df.get("Tipo General", "").eq("Desahorro")
        if "Tipo General" in df.columns
        else pd.Series(False, index=df.index)
    )
    es_neg = df["Importe"] < 0
    es_pos = df["Importe"] > 0
    df["Ahorro"] = np.where(es_ahorro, -df["Importe"], 0.0)
    df["Gasto"] = np.where(es_gasto, -df["Importe"], 0.0)
    df["Ingreso"] = np.select(
        [es_ingreso, es_desahorro],  # condición 1  # condición 2
        [
            df["Importe"],  # si es ingreso → positivo
            -df["Importe"],  # si es desahorro → negativo
        ],
        default=0.0,  # si no cumple ninguna
    )

    # Calendario
    df["Año"] = df["Fecha"].dt.year
    df["Mes"] = df["Fecha"].dt.month
    df["DiaSemanaNum"] = df["Fecha"].dt.weekday  # 0=Lun...6=Dom
    dias_es = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
    df["DiaSemana"] = df["DiaSemanaNum"].map(lambda i: dias_es[i])

    return df
