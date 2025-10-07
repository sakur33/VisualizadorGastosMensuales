#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualización de gastos personales a partir de un CSV con columnas:
Fecha,Tipo,Categoría,Concepto,Import

Requisitos:
- Python 3.9+
- pandas, matplotlib, numpy

Uso:
    python visualizar_gastos.py --csv gastos_sep_2025.csv --salidas ./salidas
"""
import shutil
import argparse
import os
import json
import unicodedata
import re
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative as pq


def leer_y_normalizar(xls_path: str) -> pd.DataFrame:
    df = pd.read_excel(xls_path, header=4, parse_dates=["F.Valor", "Fecha"])
    with open("categorias_de_gasto.json", "r", encoding="utf-8") as f:
        raw_rules = json.load(f)
    rules = _preprocess_rules(raw_rules)

    df[["Tipo", "Tipo General"]] = df.apply(
        categorizar_gastos,
        axis=1,
        args=(rules,),
        result_type="expand",
    )
    df = df.drop(
        columns=[col for col in df.columns if "Unnamed" in str(col)], errors="ignore"
    )
    df["F.Valor"] = pd.to_datetime(df["F.Valor"], dayfirst=True, errors="coerce")
    df["Fecha"] = pd.to_datetime(df["Fecha"], dayfirst=True, errors="coerce")

    # Convierte Importe a float
    df["Importe"] = pd.to_numeric(df["Importe"], errors="coerce")
    df = df.dropna(subset=["Fecha", "Importe"]).copy()

    # Limpieza básica de strings
    for col in ["Tipo", "Categoría", "Concepto"]:
        if col in df.columns:
            df[col] = df[col].fillna("").str.strip()

    es_ahorro = df["Tipo"].eq("Ahorro")
    neg = df["Importe"] < 0
    pos = df["Importe"] > 0

    # Ahorro: cuando sale dinero a la hucha (movimiento negativo etiquetado como Ahorro)
    df["Ahorro"] = np.where(es_ahorro & neg, -df["Importe"], 0.0)

    # Gasto: movimientos negativos que NO son ahorro
    df["Gasto"] = np.where(neg & ~es_ahorro, -df["Importe"], 0.0)

    # (Opcional) Ingreso: movimientos positivos que NO son ahorro
    df["Ingreso"] = np.where(pos & ~es_ahorro, df["Importe"], 0.0)

    # Día, semana, mes para vistas
    df["Año"] = df["Fecha"].dt.year
    df["Mes"] = df["Fecha"].dt.month
    df["DiaSemanaNum"] = df["Fecha"].dt.weekday  # 0=Lunes ... 6=Domingo
    dias_es = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
    df["DiaSemana"] = df["DiaSemanaNum"].map(lambda i: dias_es[i])

    return df


def _preprocess_rules(raw_rules):
    """Admite lista de dicts ({pattern_substring, Tipo}) o lista de (pat, tipo)."""
    out = []
    for r in raw_rules:
        if isinstance(r, dict):
            pat = _norm(r.get("pattern_substring", ""))
            tipo = str(r.get("Tipo", "(Sin tipo)")).strip()
            general = str(r.get("General", "(Sin tipo)"))
        else:
            pat = _norm(r[0])
            tipo = r[1]
            general = r[2]
        if pat:
            out.append((pat, tipo, general))
    return out


def _norm(s: str) -> str:
    s = str(s or "")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def categorizar_gastos(row, rules):
    """
    Devuelve (tipo, tipo_general) usando el patrón que más caracteres aporta
    entre todos los que aparecen en el texto normalizado.
    Empate → mantiene el primero que aparezca en rules.
    """
    texto = _norm(f"{row.get('Concepto', '')} {row.get('Movimiento', '')}")
    mejor = ("Sin categorizar", "Sin categorizar")
    mejor_len = -1

    for pat, tipo, tipo_general in rules:
        p = _norm(pat)
        if p and p in texto and len(p) > mejor_len:
            mejor = (tipo, tipo_general)
            mejor_len = len(p)

    return mejor


def assign_tipo(concepto: str, rules_list) -> str:
    text = (concepto or "").lower()
    for r in rules_list:  # se respeta el orden del JSON
        pat = str(r.get("pattern_substring", "")).lower()
        if pat and pat in text:
            return r.get("Tipo", "(Sin tipo)")
    return "(Sin tipo)"


def asegurar_dir(out_dir: str, str_fechas: str) -> Path:
    p = Path(out_dir + "_" + str_fechas)
    p.mkdir(parents=True, exist_ok=True)
    return p


def formato_euros(valor: float) -> str:
    return f"{valor:,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")


def chart_guardar(fig, out_path: Path, nombre: str):
    html_path = out_path / f"{nombre}.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    print(f"✅ Guardado: {html_path}")


def vista_resumen_mes(df: pd.DataFrame, out_dir: Path, str_fechas: str):
    # --------- Totales y CSV (igual que antes) ----------
    total_gastos = (
        df[df["Tipo"] != "Ahorro"]["Gasto"].sum() if "Gasto" in df.columns else 0.0
    )
    total_ingresos = df["Ingreso"].sum() if "Ingreso" in df.columns else 0.0
    ahorro = df["Ahorro"].sum() if "Ingreso" in df.columns else 0.0
    neto = total_ingresos - total_gastos
    ahorro_rate = 0.0 if total_ingresos == 0 else (neto / total_ingresos) * 100

    resumen = pd.DataFrame(
        {
            "Métrica": [
                "Ingresos",
                "Gastos",
                "Neto",
                "Ahorro",
                "% Ahorro sobre ingresos",
            ],
            "Valor": [total_ingresos, total_gastos, neto, ahorro, ahorro_rate],
            "Formateado": [
                formato_euros(total_ingresos),
                formato_euros(total_gastos),
                formato_euros(neto),
                formato_euros(ahorro),
                f"{ahorro_rate:.1f} %",
            ],
        }
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    resumen.to_csv(out_dir / "resumen_mes.csv", index=False)
    print("✅ Guardado: resumen_mes.csv")

    # --------- Preparación datos apilados ----------
    cats = ["Ingreso", "Gasto", "Ahorro"]

    by_tipo_ing = (
        df.groupby("Tipo")["Ingreso"].sum()
        if {"Tipo", "Ingreso"} <= set(df.columns)
        else pd.Series(dtype=float)
    )
    by_tipo_gas = (
        df.groupby("Tipo")["Gasto"].sum()
        if {"Tipo", "Gasto"} <= set(df.columns)
        else pd.Series(dtype=float)
    )
    by_tipo_aho = (
        df.groupby("Tipo")["Ahorro"].sum()
        if {"Tipo", "Ahorro"} <= set(df.columns)
        else pd.Series(dtype=float)
    )

    tipos = sorted(
        set(by_tipo_ing.index) | set(by_tipo_gas.index) | set(by_tipo_aho.index)
    )
    stack_df = pd.DataFrame(index=cats, columns=tipos, dtype=float).fillna(0.0)
    for t in tipos:
        stack_df.loc["Ingreso", t] = float(by_tipo_ing.get(t, 0.0))
        stack_df.loc["Gasto", t] = float(by_tipo_gas.get(t, 0.0))
        stack_df.loc["Ahorro", t] = float(by_tipo_aho.get(t, 0.0))

    # Totales por categoría (para % en tooltip)
    totales_cat = stack_df.sum(axis=1).replace(0, np.nan)  # evitar division por 0

    # --------- Colores y figura ----------
    palette = pq.Set3 + pq.Set2 + pq.Set1 + pq.Pastel1 + pq.Pastel2
    fig = go.Figure()
    x_vals = stack_df.index.tolist()

    for i, tipo in enumerate(stack_df.columns):
        y_vals = stack_df[tipo].values.astype(float)

        # % por categoría para tooltip
        pct_vals = []
        for xi, y in zip(x_vals, y_vals):
            tot = float(totales_cat.get(xi, np.nan))
            pct_vals.append(0.0 if (np.isnan(tot) or tot == 0.0) else (y / tot * 100.0))

        # customdata: [Tipo, %]
        custom = np.column_stack([np.array([tipo] * len(x_vals)), np.array(pct_vals)])

        fig.add_bar(
            name=tipo,
            x=x_vals,
            y=y_vals,
            marker_color=palette[i % len(palette)],
            customdata=custom,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Categoría: %{x}<br>"
                "Importe: %{y:,.2f} €<br>"
                "Peso en categoría: %{customdata[1]:.1f}%"
                "<extra></extra>"
            ),
        )

    fig.update_layout(
        title=f"Resumen del mes por categoría y tipo {str_fechas}",
        barmode="stack",
        xaxis_title="Categoría",
        yaxis_title="€",
        legend_title_text="Tipo",
        margin=dict(l=40, r=40, t=60, b=40),
    )

    # Etiquetas de total encima de cada barra
    # (añadimos una traza scatter para mostrar los totales formateados)
    tot_y = stack_df.sum(axis=1).values.astype(float)
    fig.add_scatter(
        x=x_vals,
        y=tot_y,
        mode="text",
        text=[formato_euros(v) for v in tot_y],
        textposition="top center",
        showlegend=False,
        hoverinfo="skip",
    )

    # --------- Guardados ----------
    chart_guardar(fig, out_path=out_dir, nombre="01_vista_resumen_mes")


def vista_flujo_diario(
    df: pd.DataFrame, out_dir: Path, str_fechas: str, window: int = 7
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Limpieza/fechas ---
    if "Fecha" not in df.columns:
        raise ValueError("El DataFrame debe tener columna 'Fecha'.")
    dfx = df.copy()
    dfx["Fecha"] = pd.to_datetime(dfx["Fecha"]).dt.normalize()

    # Garantiza columnas numéricas (o las crea a 0 si no existen)
    for col in ["Gasto", "Ingreso", "Ahorro"]:
        if col not in dfx.columns:
            dfx[col] = 0.0
    if "Tipo" not in dfx.columns:
        dfx["Tipo"] = "Sin tipo"

    # Rango completo de fechas para evitar huecos
    all_days = pd.date_range(dfx["Fecha"].min(), dfx["Fecha"].max(), freq="D")

    # Agregados por (Fecha, Tipo)
    agg = (
        dfx.groupby(["Fecha", "Tipo"])[["Gasto", "Ingreso", "Ahorro"]]
        .sum()
        .reindex(
            pd.MultiIndex.from_product(
                [all_days, sorted(dfx["Tipo"].unique())], names=["Fecha", "Tipo"]
            ),
            fill_value=0.0,
        )
    )
    # Totales diarios (para media móvil)
    gasto_diario = agg["Gasto"].groupby(level="Fecha").sum()
    media_movil = gasto_diario.rolling(window, min_periods=1).mean()

    # Totales por categoría y día (para % en tooltip)
    tot_gasto = agg["Gasto"].groupby(level="Fecha").sum().replace(0, np.nan)
    tot_ingreso = agg["Ingreso"].groupby(level="Fecha").sum().replace(0, np.nan)
    tot_ahorro = agg["Ahorro"].groupby(level="Fecha").sum().replace(0, np.nan)

    # Colores consistentes por Tipo
    palette = pq.Set3 + pq.Set2 + pq.Set1 + pq.Pastel1 + pq.Pastel2
    tipos = sorted(dfx["Tipo"].unique())
    tipo2color = {t: palette[i % len(palette)] for i, t in enumerate(tipos)}

    fig = go.Figure()
    x_vals = all_days

    # Helper para añadir una "pila" (stack) por categoría
    def add_stacked_for_category(cat_name: str, offsetgroup: str):
        # cat_name ∈ {"Gasto","Ingreso","Ahorro"}
        # offsetgroup separa las pilas para que queden agrupadas lado a lado
        for j, tipo in enumerate(tipos):
            serie = agg.loc[(slice(None), tipo), cat_name].values.astype(float)

            # % de ese tipo dentro del total de la categoría del día
            if cat_name == "Gasto":
                totales = tot_gasto.reindex(all_days).values.astype(float)
            elif cat_name == "Ingreso":
                totales = tot_ingreso.reindex(all_days).values.astype(float)
            else:
                totales = tot_ahorro.reindex(all_days).values.astype(float)

            pct = np.where(
                (~np.isfinite(totales)) | (totales == 0.0),
                0.0,
                (serie / totales) * 100.0,
            )

            # Agrupamos por tipo en la leyenda, pero mostramos solo una vez el nombre del tipo
            show_legend = cat_name == "Ingreso"  # solo en Ingreso aparece en la leyenda
            fig.add_bar(
                name=tipo,
                legendgroup=tipo,  # al ocultar/mostrar, afecta a las tres categorías del mismo tipo
                showlegend=show_legend,
                x=x_vals,
                y=serie,
                marker_color=tipo2color[tipo],
                offsetgroup=offsetgroup,  # separa las pilas por categoría
                customdata=np.column_stack(
                    [
                        np.array([cat_name] * len(x_vals)),  # Categoría
                        np.array([tipo] * len(x_vals)),  # Tipo
                        pct,  # %
                    ]
                ),
                hovertemplate=(
                    "Fecha: %{x|%Y-%m-%d}<br>"
                    "Categoría: %{customdata[0]}<br>"
                    "Tipo: <b>%{customdata[1]}</b><br>"
                    "Importe: %{y:,.2f} €<br>"
                    "Peso en categoría: %{customdata[2]:.1f}%"
                    "<extra></extra>"
                ),
            )

    # Añadimos las tres pilas lado a lado
    add_stacked_for_category("Ingreso", offsetgroup="Ingreso")
    add_stacked_for_category("Gasto", offsetgroup="Gasto")
    add_stacked_for_category("Ahorro", offsetgroup="Ahorro")

    # --- Línea de media móvil de gasto diario ---
    fig.add_trace(
        go.Scatter(
            x=media_movil.index,
            y=media_movil.values,
            mode="lines",
            name=f"Media móvil {window} días (Gasto)",
            line=dict(color="red", width=2.5),
            hovertemplate="Fecha: %{x|%Y-%m-%d}<br>Media 7d: %{y:,.2f} €<extra></extra>",
            # yaxis="y2",  # segunda escala si queremos separarla visualmente
        )
    )

    fig.update_layout(
        title=f"Flujo diario: Ingreso vs Gasto vs Ahorro (agrupado) con apilado por tipo  {str_fechas}",
        barmode="relative",  # apilado dentro de cada offsetgroup
        bargap=0.25,  # separación entre días
        bargroupgap=0.15,  # separación entre pilas (Ingreso/Gasto/Ahorro) en el mismo día
        xaxis_title="Fecha",
        yaxis_title="€",
        legend_title_text="Tipo",
        margin=dict(l=40, r=40, t=60, b=40),
    )

    # Export interactivo
    chart_guardar(fig, out_path=out_dir, nombre="02_flujo_diario_stacked_por_tipo.html")


def vista_por_tipo(df: pd.DataFrame, out_dir: Path, str_fechas: str):
    """
    Barras horizontales apiladas por Tipo, con segmentos por Concepto (suma de Gasto).
    Tooltip: Tipo, Concepto, Importe y % dentro del Tipo.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    req_cols = {"Tipo", "Concepto", "Gasto"}
    if not req_cols.issubset(df.columns):
        raise ValueError(f"El DataFrame debe tener columnas {req_cols}.")

    dfx = df.copy()
    dfx["Gasto"] = pd.to_numeric(dfx["Gasto"], errors="coerce").fillna(0.0)
    dfx = dfx[dfx["Gasto"] > 0]  # solo gastos positivos

    # --- Agregación por Tipo x Concepto ---
    grp = dfx.groupby(["Tipo", "Concepto"])["Gasto"].sum().reset_index()

    # (Opcional) si hay demasiados conceptos, agrupa los menores como "Otros"
    n_conceptos = grp["Concepto"].nunique()
    if n_conceptos > 20:
        top_k = 20
        tot_por_concepto = (
            grp.groupby("Concepto")["Gasto"].sum().sort_values(ascending=False)
        )
        top_conceptos = set(tot_por_concepto.head(top_k).index)
        grp["Concepto"] = np.where(
            grp["Concepto"].isin(top_conceptos), grp["Concepto"], "Otros"
        )
        grp = grp.groupby(["Tipo", "Concepto"])["Gasto"].sum().reset_index()

    # Pivot: filas = Tipo, columnas = Concepto, valores = Gasto
    pivot = grp.pivot_table(
        index="Tipo", columns="Concepto", values="Gasto", aggfunc="sum", fill_value=0.0
    )

    # Ordenar Tipos por total ascendente (como tu versión original)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=True).index]

    # Totales por Tipo (para CSV y etiquetas)
    tot_por_tipo = pivot.sum(axis=1)
    por_tipo = tot_por_tipo.sort_values(ascending=True)

    # Guardar CSV de totales por Tipo (mantiene tu salida original)
    (out_dir / "gasto_por_tipo.csv").write_text(
        por_tipo.to_csv(header=["Gasto"]), encoding="utf-8"
    )
    print("✅ Guardado: gasto_por_tipo.csv")

    # % dentro de cada Tipo (para tooltip)
    share = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0) * 100.0
    share = share.fillna(0.0)

    # Colores por Concepto
    conceptos = pivot.columns.tolist()
    palette = pq.Set3 + pq.Set2 + pq.Set1 + pq.Pastel1 + pq.Pastel2
    color_map = {c: palette[i % len(palette)] for i, c in enumerate(conceptos)}

    # Figura: una traza por Concepto, apiladas horizontalmente
    fig = go.Figure()
    y_cats = pivot.index.astype(str).tolist()

    for concepto in conceptos:
        x_vals = pivot[concepto].values.astype(float)
        if np.allclose(x_vals, 0.0):  # omite conceptos sin aporte
            continue
        pct_vals = share[concepto].values
        custom = np.column_stack(
            [np.array([concepto] * len(y_cats), dtype=object), pct_vals]
        )

        fig.add_bar(
            name=str(concepto),
            y=y_cats,
            x=x_vals,
            orientation="h",
            marker_color=color_map[concepto],
            customdata=custom,
            hovertemplate=(
                "Tipo: %{y}<br>"
                "Concepto: <b>%{customdata[0]}</b><br>"
                "Gasto: %{x:,.2f} €<br>"
                "Peso en el tipo: %{customdata[1]:.1f}%"
                "<extra></extra>"
            ),
        )

    # Etiqueta de total al final de cada barra
    max_total = float(tot_por_tipo.max() or 0.0)
    fig.add_scatter(
        y=y_cats,
        x=tot_por_tipo.values.astype(float),
        mode="text",
        text=[f"{v:,.0f} €" for v in tot_por_tipo.values],
        textposition="middle right",
        textfont=dict(size=11),
        showlegend=False,
        hoverinfo="skip",
    )

    fig.update_layout(
        title=f"Gasto por Tipo (apilado por Concepto)  {str_fechas}",
        xaxis_title="€",
        yaxis_title="Tipo",
        barmode="stack",
        margin=dict(l=110, r=40, t=70, b=40),
        height=max(380, len(y_cats) * 32),
        legend_title_text="Concepto",
    )

    # Deja espacio para la etiqueta de total a la derecha
    fig.update_xaxes(range=[0, max_total * 1.12 if max_total > 0 else 1])

    # Guardar HTML con tu helper
    chart_guardar(fig, out_path=out_dir, nombre="03_gasto_por_tipo.html")


def vista_por_tipo_general(df: pd.DataFrame, out_dir: Path, str_fechas: str):
    """
    Genera un gráfico horizontal (barh) de gasto total por tipo en Plotly,
    con tooltips y etiquetas de valor.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if "Tipo General" not in df.columns or "Gasto" not in df.columns:
        raise ValueError("El DataFrame debe tener columnas 'Tipo' y 'Gasto'.")

    # --- Agregación y orden ---
    por_tipo = df.groupby("Tipo General")["Gasto"].sum().sort_values(ascending=True)
    total_gasto = por_tipo.sum()
    porcentajes = (por_tipo / total_gasto * 100).round(1)

    # --- Guardar CSV ---
    csv_path = out_dir / "gasto_por_tipo.csv"
    por_tipo.to_csv(csv_path, header=["Gasto"])
    print(f"✅ Guardado: {csv_path.name}")

    # --- Colores y configuración ---
    palette = pq.Set3 + pq.Set2 + pq.Set1 + pq.Pastel1 + pq.Pastel2
    colores = [palette[i % len(palette)] for i in range(len(por_tipo))]

    # --- Figura Plotly ---
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=por_tipo.values,
            y=por_tipo.index.astype(str),
            orientation="h",
            marker_color=colores,
            customdata=porcentajes.values,
            text=[f"{v:,.0f} €" for v in por_tipo.values],
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Gasto: %{x:,.2f} €<br>"
                "Peso sobre total: %{customdata:.1f}%"
                "<extra></extra>"
            ),
        )
    )

    # --- Layout general ---
    fig.update_layout(
        title=f"Gasto por Tipo General  {str_fechas}",
        xaxis_title="€",
        yaxis_title="Tipo",
        margin=dict(l=100, r=40, t=60, b=40),
        height=max(350, len(por_tipo) * 30),
        showlegend=False,
    )

    # --- Guardar usando tu función chart_guardar ---
    chart_guardar(fig, out_path=out_dir, nombre="04_gasto_por_tipo_general.html")


def vista_top_gastos(df: pd.DataFrame, out_dir: Path, str_fechas: str, top_n: int = 10):
    """
    Muestra el Top-N de categorías (comercios) por Gasto con barras horizontales interactivas.
    Guarda CSV y HTML (vía chart_guardar).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if "Concepto" not in df.columns or "Gasto" not in df.columns:
        raise ValueError("El DataFrame debe tener columnas 'Categoría' y 'Gasto'.")

    por_cat = (
        df.groupby("Concepto")["Gasto"].sum().sort_values(ascending=False).head(top_n)
    )

    total_top = por_cat.sum()
    porcentajes = (por_cat / total_top * 100).round(1)

    # Guardar CSV igual que antes
    csv_path = out_dir / "top_gastos.csv"
    por_cat.to_csv(csv_path, header=["Gasto"])
    print(f"✅ Guardado: {csv_path.name}")

    # Orden visual: mayor arriba → invertimos para barh
    cats = por_cat.index.astype(str)[::-1]
    vals = por_cat.values[::-1]
    pct_vals = porcentajes.values[::-1]

    palette = pq.Set3 + pq.Set2 + pq.Set1 + pq.Pastel1 + pq.Pastel2
    colores = [palette[i % len(palette)] for i in range(len(cats))]

    fig = go.Figure(
        go.Bar(
            x=vals,
            y=cats,
            orientation="h",
            marker_color=colores,
            customdata=pct_vals,
            text=[f"{v:,.0f} €" for v in vals],
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Gasto: %{x:,.2f} €<br>"
                "Peso en Top: %{customdata:.1f}%"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=f"Top {top_n} gastos  {str_fechas}",
        xaxis_title="€",
        yaxis_title="Categoría",
        margin=dict(l=120, r=40, t=60, b=40),
        height=max(350, len(cats) * 32),
        showlegend=False,
    )

    chart_guardar(fig, out_path=out_dir, nombre="05_top_gastos.html")


def vista_por_semana(df: pd.DataFrame, out_dir: Path, str_fechas: str):
    """
    Gráfico interactivo (Plotly) del gasto semanal (Lunes→Domingo),
    con etiquetas, tooltip y guardado automático.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if "Fecha" not in df.columns or "Gasto" not in df.columns:
        raise ValueError("El DataFrame debe tener columnas 'Fecha' y 'Gasto'.")

    # Normalizamos fecha y calculamos el lunes de cada semana
    df["Fecha"] = pd.to_datetime(df["Fecha"]).dt.normalize()
    df["Semana_Lunes"] = df["Fecha"] - pd.to_timedelta(df["Fecha"].dt.weekday, unit="D")

    por_semana = df.groupby("Semana_Lunes")["Gasto"].sum().sort_index()

    # Guardar CSV
    csv_path = out_dir / "gasto_por_semana.csv"
    por_semana.rename("Gasto").to_csv(csv_path, header=True)
    print(f"✅ Guardado: {csv_path.name}")

    # Colores
    palette = pq.Set3 + pq.Set2 + pq.Set1 + pq.Pastel1 + pq.Pastel2
    color = palette[0]

    # Eje X como fechas (semanas)
    x_vals = por_semana.index
    y_vals = por_semana.values

    # Figura
    fig = go.Figure(
        go.Bar(
            x=x_vals,
            y=y_vals,
            marker_color=color,
            text=[f"{v:,.0f} €" for v in y_vals],
            textposition="outside",
            hovertemplate=(
                "Semana que empieza: %{x|%Y-%m-%d}<br>"
                "Gasto total: %{y:,.2f} €<extra></extra>"
            ),
        )
    )

    # Layout
    fig.update_layout(
        title=f"Gasto por semana (Lun→Dom)  {str_fechas}",
        xaxis_title="Semana (lunes)",
        yaxis_title="€",
        margin=dict(l=60, r=40, t=60, b=60),
        height=400,
        showlegend=False,
    )

    # Rotar etiquetas de fechas si hay muchas semanas
    fig.update_xaxes(tickangle=-45)

    # Guardar con tu función común
    chart_guardar(fig, out_path=out_dir, nombre="06_gasto_por_semana.html")


def vista_media_por_dia_semana(
    df: pd.DataFrame,
    out_dir: Path,
    str_fechas: str,
    incluir_dias_sin_gasto: bool = True,
):
    """
    Un solo gráfico:
      - Barras (eje izq): TOTAL gastado por día de la semana
      - Línea  (eje der): MEDIA diaria por ese día
    Las etiquetas no se pisan: barras -> arriba (outside), línea -> debajo del punto.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if "Gasto" not in df.columns:
        raise ValueError("El DataFrame debe tener columna 'Gasto'.")

    # Usar F.Valor si existe, si no Fecha
    date_col = (
        "F.Valor"
        if "F.Valor" in df.columns
        else ("Fecha" if "Fecha" in df.columns else None)
    )
    if date_col is None:
        raise ValueError("Se requiere 'F.Valor' o 'Fecha' en el DataFrame.")

    dfx = df.copy()
    dfx[date_col] = pd.to_datetime(dfx[date_col])
    dfx["Fecha"] = dfx[date_col].dt.normalize()
    dfx["Gasto"] = pd.to_numeric(dfx["Gasto"], errors="coerce").fillna(0.0)

    # 1) gasto DIARIO (sumar movimientos por día)
    gasto_diario = dfx.groupby("Fecha")["Gasto"].sum()

    # 2) opcional: incluir días sin gasto para que media * nº días ≈ total
    if incluir_dias_sin_gasto and not gasto_diario.empty:
        all_days = pd.date_range(
            gasto_diario.index.min(), gasto_diario.index.max(), freq="D"
        )
        gasto_diario = gasto_diario.reindex(all_days, fill_value=0.0)

    # 3) agregación por día de semana
    di = gasto_diario.to_frame("Gasto")
    di["DiaSemanaNum"] = di.index.weekday  # 0=Lun..6=Dom
    agg = di.groupby("DiaSemanaNum")["Gasto"].agg(
        media="mean", total="sum", n_dias="size"
    )

    nombres = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    agg = agg.reindex(range(7))
    agg.insert(0, "DiaSemana", nombres)

    # CSV
    csv_path = out_dir / "media_y_total_gasto_por_dia_semana.csv"
    agg.to_csv(csv_path, index=False)
    print(f"✅ Guardado: {csv_path.name}")

    # Colores
    palette = pq.Set3 + pq.Set2 + pq.Set1 + pq.Pastel1 + pq.Pastel2
    color_bar = palette[0]
    color_line = "#E74C3C"

    # Márgenes para que no se corten textos
    total_max = float(agg["total"].max() or 0)
    media_max = float(agg["media"].max() or 0)

    fig = go.Figure()

    # Barras (TOTAL, etiquetas arriba)
    fig.add_bar(
        x=agg["DiaSemana"],
        y=agg["total"],
        name="Total",
        marker_color=color_bar,
        text=[f"{v:,.0f} €" for v in agg["total"].values],
        textposition="outside",
        cliponaxis=False,  # permite que el texto salga del área y no se corte
        customdata=np.column_stack([agg["media"].values, agg["n_dias"].values]),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Total: %{y:,.2f} €<br>"
            "Media diaria: %{customdata[0]:,.2f} €<br>"
            "Nº de días: %{customdata[1]:,d}"
            "<extra></extra>"
        ),
    )

    # Línea (MEDIA, etiquetas debajo del punto para no pisar las de barra)
    fig.add_scatter(
        x=agg["DiaSemana"],
        y=agg["media"],
        name="Media diaria",
        mode="lines+markers+text",
        line=dict(color=color_line, width=3),
        text=[f"{v:,.0f} €" for v in agg["media"].values],
        textposition="bottom center",  # <- clave para no pisar las de barra
        textfont=dict(size=11),
        cliponaxis=False,
        hovertemplate="<b>%{x}</b><br>Media diaria: %{y:,.2f} €<extra></extra>",
        yaxis="y2",  # eje secundario superpuesto
    )

    fig.update_layout(
        title=f"Gasto por día de la semana: Total (barras) y Media (línea)  {str_fechas}",
        xaxis_title="Día de la semana",
        yaxis=dict(
            title="€ (total)",
            range=[0, total_max * 1.18] if total_max > 0 else [0, 1],
        ),
        yaxis2=dict(
            title="€ (media diaria)",
            overlaying="y",
            side="right",
            showgrid=False,
            range=(
                [0, media_max * 1.8] if media_max > 0 else [0, 1]
            ),  # un pelín más alto para el texto abajo
        ),
        margin=dict(l=60, r=70, t=60, b=60),
        legend_title_text="Métrica",
        hovermode="x unified",
        bargap=0.25,
    )

    chart_guardar(
        fig, out_path=out_dir, nombre="07_media_y_total_gasto_por_dia_semana.html"
    )


def _etiqueta_gasto(row):
    # etiqueta compacta:  DD/MM  ·  Concepto|Categoría
    fecha = row["Fecha"].strftime("%d/%m")
    base = row["Concepto"] if row.get("Concepto", "") else row.get("Categoría", "")
    txt = f"{fecha} · {base}".strip(" ·")
    # recorta etiquetas muy largas
    return (txt[:40] + "…") if len(txt) > 41 else txt


def vista_barras_por_tipo_detalle(
    df: pd.DataFrame, out_dir: Path, str_fechas: str, min_barras=1
):
    carpeta = out_dir / "por_tipo"
    carpeta.mkdir(parents=True, exist_ok=True)

    # Fecha robusta: usa 'Fecha' o 'F.Valor'
    if "Fecha" in df.columns:
        fechas = pd.to_datetime(df["Fecha"], errors="coerce")
    elif "F.Valor" in df.columns:
        fechas = pd.to_datetime(df["F.Valor"], errors="coerce")
    else:
        raise ValueError("Se requiere columna 'Fecha' o 'F.Valor'.")

    dfx = df.copy()
    dfx["Fecha"] = fechas.dt.normalize()
    if "Tipo" not in dfx.columns:
        dfx["Tipo"] = "Sin tipo"

    tipos = sorted([t for t in dfx["Tipo"].dropna().unique() if str(t).strip() != ""])
    palette = pq.Set3 + pq.Set2 + pq.Set1 + pq.Pastel1 + pq.Pastel2

    for idx, tipo in enumerate(tipos):
        dft = dfx[dfx["Tipo"] == tipo].copy()
        dft["Gasto"] = pd.to_numeric(dft.get("Gasto", 0.0), errors="coerce").fillna(0.0)
        dft = dft[dft["Gasto"] > 0]  # solo gastos positivos
        if dft.empty or len(dft) < min_barras:
            continue

        # Etiqueta compacta (usa tu helper)
        dft["etiqueta"] = dft.apply(_etiqueta_gasto, axis=1)
        dft = dft.sort_values("Gasto", ascending=False)

        color = palette[idx % len(palette)]
        width = max(900, min(1800, 22 * len(dft) + 600))  # ancho dinámico

        fig = go.Figure(
            go.Bar(
                x=dft["etiqueta"],
                y=dft["Gasto"],
                marker_color=color,
                text=[formato_euros(v) for v in dft["Gasto"]],
                textposition="outside",
                cliponaxis=False,  # deja que el texto sobresalga sin cortarse
                customdata=np.column_stack(
                    [
                        pd.to_datetime(dft["Fecha"]).dt.strftime("%Y-%m-%d"),
                        dft.get("Concepto", pd.Series([""] * len(dft))).astype(str),
                        dft.get("Categoría", pd.Series([""] * len(dft))).astype(str),
                    ]
                ),
                hovertemplate=(
                    "Fecha: %{customdata[0]}<br>"
                    "Concepto: %{customdata[1]}<br>"
                    "Categoría: %{customdata[2]}<br>"
                    "Importe: %{y:,.2f} €"
                    "<extra></extra>"
                ),
            )
        )

        fig.update_layout(
            title=f"Gastos de “{tipo}” por operación  {str_fechas}",
            xaxis_title="Operación (fecha · concepto)",
            yaxis_title="€",
            xaxis=dict(tickangle=-75),
            margin=dict(l=60, r=40, t=60, b=100),
            showlegend=False,
        )

        # Nombre de archivo seguro
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", str(tipo)).strip("_")
        chart_guardar(fig, out_path=carpeta, nombre=f"tipo_{slug}")


def vista_histograma_gastos(
    df: pd.DataFrame, out_dir: Path, str_fechas: str, nbins: int = 20
):
    # Filtra solo gastos positivos y numéricos
    if "Gasto" not in df.columns:
        raise ValueError("El DataFrame debe tener columna 'Gasto'.")
    gastos = pd.to_numeric(df["Gasto"], errors="coerce")
    gastos = gastos[gastos > 0].dropna()
    if gastos.empty:
        return

    # Estadísticos
    mean_val = gastos.mean()
    median_val = gastos.median()

    # Calculamos histograma manualmente para disponer de los bordes de cada bin
    counts, edges = np.histogram(gastos.values, bins=nbins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    widths = edges[1:] - edges[:-1]

    # Color
    palette = pq.Set3 + pq.Set2 + pq.Set1 + pq.Pastel1 + pq.Pastel2
    bar_color = palette[2]

    # Figura como barras (no trace Histogram) para controlar el tooltip con el rango
    fig = go.Figure(
        go.Bar(
            x=centers,
            y=counts,
            width=widths,
            marker_color=bar_color,
            customdata=np.column_stack([edges[:-1], edges[1:]]),
            hovertemplate=(
                "Rango: %{customdata[0]:,.2f}–%{customdata[1]:,.2f} €<br>"
                "Frecuencia: %{y:,d}"
                "<extra></extra>"
            ),
        )
    )

    # Líneas de referencia: media (dash) y mediana (dot)
    fig.add_vline(x=mean_val, line_dash="dash", line_width=2, line_color="#2E86C1")
    fig.add_vline(x=median_val, line_dash="dot", line_width=2, line_color="#E74C3C")

    # Anotaciones arriba
    fig.add_annotation(
        x=mean_val,
        y=1.08,
        yref="paper",
        showarrow=False,
        text=f"Media {mean_val:,.2f} €",
        font=dict(size=11),
        xanchor="center",
    )
    fig.add_annotation(
        x=median_val,
        y=1.15,
        yref="paper",
        showarrow=False,
        text=f"Mediana {median_val:,.2f} €",
        font=dict(size=11),
        xanchor="center",
    )

    fig.update_layout(
        title=f"Distribución de importes de gasto  {str_fechas}",
        xaxis_title="€ por operación (rango del bin)",
        yaxis_title="Frecuencia",
        bargap=0.05,
        margin=dict(l=60, r=40, t=70, b=50),
    )

    chart_guardar(fig, out_path=out_dir, nombre="09_histograma_gastos.html")


def exportar_tablas_utiles(df: pd.DataFrame, out_dir: Path, str_fechas: str):
    # Tablas reutilizables en Excel/Sheets
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


def main():
    parser = argparse.ArgumentParser(description="Visualizar gastos personales.")
    parser.add_argument(
        "--xls",
        dest="xls_path",
        default="./2025Y-10M-06D-15_37_11-Últimos movimientos.xlsx",
        help="Ruta al CSV (p. ej. gastos_sep_2025.csv)",
    )
    parser.add_argument(
        "--salidas",
        dest="out_dir",
        default="./salidas",
        help="Carpeta de salida (por defecto ./salidas)",
    )
    args = parser.parse_args()
    if "test" in args.xls_path:
        print("Runing a test file")
    df = leer_y_normalizar(args.xls_path)

    FMT = "%Y-%m-%d"
    FECHA_INICIO = df["F.Valor"].min().strftime(FMT)
    FECHA_FIN = df["F.Valor"].max().strftime(FMT)
    str_fechas = f"{FECHA_INICIO} - {FECHA_FIN}"

    out_dir = asegurar_dir(args.out_dir, str_fechas)

    # Vistas / gráficos
    vista_resumen_mes(df, out_dir, str_fechas=str_fechas)
    vista_flujo_diario(df, out_dir, str_fechas=str_fechas)
    vista_por_tipo(df, out_dir, str_fechas=str_fechas)
    vista_por_tipo_general(df, out_dir, str_fechas=str_fechas)
    vista_top_gastos(df, out_dir, top_n=10, str_fechas=str_fechas)
    vista_por_semana(df, out_dir, str_fechas=str_fechas)
    vista_media_por_dia_semana(df, out_dir, str_fechas=str_fechas)
    vista_barras_por_tipo_detalle(df, out_dir, str_fechas=str_fechas)
    vista_histograma_gastos(df, out_dir, str_fechas=str_fechas)

    # CSVs auxiliares
    exportar_tablas_utiles(df, out_dir, str_fechas=str_fechas)

    print("\nListo. Abre la carpeta de salidas y revisa los PNG/CSV generados.")

    destino = Path("/mnt/c/Users/alejandro/Downloads")
    if destino.exists():
        nuevo_path = destino / f"salidas_{str_fechas}"
        # remove old folder if exists
        if nuevo_path.exists():
            shutil.rmtree(nuevo_path)
        try:
            shutil.move(str(out_dir), str(nuevo_path))
            print(f"📦 Carpeta movida correctamente a: {nuevo_path}")
        except Exception as e:
            print(
                f"⚠️ No se pudo mover la carpeta automáticamente ({e}). "
                f"Muévela manualmente con:\n  mv {out_dir} {nuevo_path}"
            )
    else:
        print(
            "⚠️ No se encontró /mnt/c/Users/alejandro/Downloads — no se movió la carpeta."
        )


if __name__ == "__main__":
    main()
