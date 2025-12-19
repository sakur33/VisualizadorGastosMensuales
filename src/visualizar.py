# /src/visualizar.py

import re
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative as pq
import utils as utils


def vista_resumen_mes(df: pd.DataFrame, out_dir: Path, str_fechas: str, top_k: int = 12):
    """
    Resumen del mes con 3 barras (Ingreso, Gasto, Ahorro) apiladas por Concepto+Movimiento.
    Tooltip incluye: Métrica, Tipo, Concepto, Movimiento, Importe.
    Guarda 'resumen_mes.csv' como antes.
    """
    # ---- Totales y CSV (igual que antes) ----
    total_gastos = df[df["Tipo"] != "Ahorro"]["Gasto"].sum() if "Gasto" in df.columns else 0.0
    total_ingresos = df["Ingreso"].sum() if "Ingreso" in df.columns else 0.0
    ahorro = df["Ahorro"].sum() if "Ahorro" in df.columns else 0.0
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
                utils.formato_euros(total_ingresos),
                utils.formato_euros(total_gastos),
                utils.formato_euros(neto),
                utils.formato_euros(ahorro),
                f"{ahorro_rate:.1f} %",
            ],
        }
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    resumen.to_csv(out_dir / "resumen_mes.csv", index=False)
    print("✅ Guardado: resumen_mes.csv")

    # ---- Requisitos para el gráfico ----
    req = {"Ingreso", "Gasto", "Ahorro", "Tipo"}
    faltan = req - set(df.columns)
    if faltan:
        raise ValueError(f"Faltan columnas para el gráfico apilado: {faltan}")

    dfx = df.copy()
    dfx["Tipo"] = dfx["Tipo"].fillna("").replace("", "Sin tipo")

    # Paleta y colores consistentes por Tipo
    tipos = sorted([t for t in dfx["Tipo"].unique() if str(t).strip() != ""])
    palette = pq.Set3 + pq.Set2 + pq.Set1 + pq.Pastel1 + pq.Pastel2
    color_by_tipo = {t: palette[i % len(palette)] for i, t in enumerate(tipos)}

    fig = go.Figure()
    x_cats = ["Ingreso", "Gasto", "Ahorro"]

    # Para cada métrica apilamos por Tipo
    for met in x_cats:
        serie = dfx.groupby("Tipo")[met].sum()
        for tipo in tipos:
            val = float(serie.get(tipo, 0.0))
            if np.isclose(val, 0.0):
                continue
            fig.add_bar(
                name=tipo,
                x=[met],
                y=[val],
                marker_color=color_by_tipo[tipo],
                offsetgroup=met,  # barras lado a lado por métrica
                legendgroup=tipo,
                showlegend=(met == "Gasto"),  # leyenda una sola vez
                customdata=np.array([[met, tipo]], dtype=object),
                hovertemplate=(
                    "Métrica: %{customdata[0]}<br>"
                    "Tipo: %{customdata[1]}<br>"
                    "Importe: %{y:,.2f} €<extra></extra>"
                ),
            )

    # Etiquetas de total encima de cada barra (Ingreso/Gasto/Ahorro)
    tot_ing = float(dfx["Ingreso"].sum()) if "Ingreso" in dfx.columns else 0.0
    tot_gas = float(dfx["Gasto"].sum()) if "Gasto" in dfx.columns else 0.0
    tot_aho = float(dfx["Ahorro"].sum()) if "Ahorro" in dfx.columns else 0.0
    fig.add_scatter(
        x=x_cats,
        y=[tot_ing, tot_gas, tot_aho],
        mode="text",
        text=[utils.formato_euros(t) for t in [tot_ing, tot_gas, tot_aho]],
        textposition="top center",
        showlegend=False,
        hoverinfo="skip",
    )

    fig.update_layout(
        title=f"Resumen del mes (Ingreso/Gasto/Ahorro apilado por Tipo)  {str_fechas}",
        barmode="relative",
        bargroupgap=0.18,
        xaxis_title="Métrica",
        yaxis_title="€",
        legend_title_text="Tipo",
        margin=dict(l=50, r=40, t=70, b=40),
    )

    utils.chart_guardar(fig, out_path=out_dir, nombre="01_vista_resumen_mes")


def vista_flujo_diario(df: pd.DataFrame, out_dir: Path, str_fechas: str, window: int = 7):
    out_dir.mkdir(parents=True, exist_ok=True)
    if "Fecha" not in df.columns:
        raise ValueError("El DataFrame debe tener columna 'Fecha'.")
    dfx = df.copy()
    dfx["Fecha"] = pd.to_datetime(dfx["Fecha"]).dt.normalize()

    for col in ["Gasto", "Ingreso", "Ahorro"]:
        if col not in dfx.columns:
            dfx[col] = 0.0
    if "Tipo" not in dfx.columns:
        dfx["Tipo"] = "Sin tipo"

    all_days = pd.date_range(dfx["Fecha"].min(), dfx["Fecha"].max(), freq="D")
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
    gasto_diario = agg["Gasto"].groupby(level="Fecha").sum()
    media_movil = gasto_diario.rolling(window, min_periods=1).mean()

    tot_gasto = agg["Gasto"].groupby(level="Fecha").sum().replace(0, np.nan)
    tot_ingreso = agg["Ingreso"].groupby(level="Fecha").sum().replace(0, np.nan)
    tot_ahorro = agg["Ahorro"].groupby(level="Fecha").sum().replace(0, np.nan)

    palette = pq.Set3 + pq.Set2 + pq.Set1 + pq.Pastel1 + pq.Pastel2
    tipos = sorted(dfx["Tipo"].unique())
    tipo2color = {t: palette[i % len(palette)] for i, t in enumerate(tipos)}

    fig = go.Figure()
    x_vals = all_days

    def add_stacked_for_category(cat_name: str, offsetgroup: str):
        for j, tipo in enumerate(tipos):
            serie = agg.loc[(slice(None), tipo), cat_name].values.astype(float)
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
            show_legend = cat_name == "Ingreso"
            fig.add_bar(
                name=tipo,
                legendgroup=tipo,
                showlegend=show_legend,
                x=x_vals,
                y=serie,
                marker_color=tipo2color[tipo],
                offsetgroup=offsetgroup,
                customdata=np.column_stack(
                    [
                        np.array([cat_name] * len(x_vals)),
                        np.array([tipo] * len(x_vals)),
                        pct,
                    ]
                ),
                hovertemplate=(
                    "Fecha: %{x|%Y-%m-%d}<br>"
                    "Categoría: %{customdata[0]}<br>"
                    "Tipo: <b>%{customdata[1]}</b><br>"
                    "Importe: %{y:,.2f} €<br>"
                    "Peso en categoría: %{customdata[2]:.1f}%<extra></extra>"
                ),
            )

    add_stacked_for_category("Ingreso", offsetgroup="Ingreso")
    add_stacked_for_category("Gasto", offsetgroup="Gasto")
    add_stacked_for_category("Ahorro", offsetgroup="Ahorro")

    fig.add_trace(
        go.Scatter(
            x=media_movil.index,
            y=media_movil.values,
            mode="lines",
            name=f"Media móvil {window} días (Gasto)",
            line=dict(color="red", width=2.5),
            hovertemplate="Fecha: %{x|%Y-%m-%d}<br>Media 7d: %{y:,.2f} €<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Flujo diario: Ingreso vs Gasto vs Ahorro (agrupado) con apilado por tipo  {str_fechas}",
        barmode="relative",
        bargap=0.25,
        bargroupgap=0.15,
        xaxis_title="Fecha",
        yaxis_title="€",
        legend_title_text="Tipo",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    utils.chart_guardar(fig, out_path=out_dir, nombre="02_flujo_diario_stacked_por_tipo")


def vista_por_tipo(df: pd.DataFrame, out_dir: Path, str_fechas: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    req_cols = {"Tipo", "Concepto", "Gasto"}
    if not req_cols.issubset(df.columns):
        raise ValueError(f"El DataFrame debe tener columnas {req_cols}.")

    dfx = df.copy()
    dfx["Gasto"] = pd.to_numeric(dfx["Gasto"], errors="coerce").fillna(0.0)
    dfx = dfx[dfx["Gasto"] > 0]

    grp = dfx.groupby(["Tipo", "Concepto"])["Gasto"].sum().reset_index()
    n_conceptos = grp["Concepto"].nunique()
    if n_conceptos > 20:
        top_k = 20
        tot_por_concepto = grp.groupby("Concepto")["Gasto"].sum().sort_values(ascending=False)
        top_conceptos = set(tot_por_concepto.head(top_k).index)
        grp["Concepto"] = np.where(grp["Concepto"].isin(top_conceptos), grp["Concepto"], "Otros")
        grp = grp.groupby(["Tipo", "Concepto"])["Gasto"].sum().reset_index()

    pivot = grp.pivot_table(
        index="Tipo", columns="Concepto", values="Gasto", aggfunc="sum", fill_value=0.0
    )
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=True).index]
    tot_por_tipo = pivot.sum(axis=1)
    por_tipo = tot_por_tipo.sort_values(ascending=True)

    (out_dir / "gasto_por_tipo.csv").write_text(por_tipo.to_csv(header=["Gasto"]), encoding="utf-8")
    print("✅ Guardado: gasto_por_tipo.csv")

    share = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0) * 100.0
    share = share.fillna(0.0)

    conceptos = pivot.columns.tolist()
    palette = pq.Set3 + pq.Set2 + pq.Set1 + pq.Pastel1 + pq.Pastel2
    color_map = {c: palette[i % len(palette)] for i, c in enumerate(conceptos)}

    fig = go.Figure()
    y_cats = pivot.index.astype(str).tolist()
    for concepto in conceptos:
        x_vals = pivot[concepto].values.astype(float)
        if np.allclose(x_vals, 0.0):
            continue
        pct_vals = share[concepto].values
        custom = np.column_stack([np.array([concepto] * len(y_cats), dtype=object), pct_vals])
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
                "Peso en el tipo: %{customdata[1]:.1f}%<extra></extra>"
            ),
        )

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
    fig.update_xaxes(range=[0, max_total * 1.12 if max_total > 0 else 1])
    utils.chart_guardar(fig, out_path=out_dir, nombre="03_gasto_por_tipo")


def vista_por_tipo_general(df: pd.DataFrame, out_dir: Path, str_fechas: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    if "Tipo General" not in df.columns or "Gasto" not in df.columns:
        raise ValueError("El DataFrame debe tener columnas 'Tipo General' y 'Gasto'.")

    por_tipo = df.groupby("Tipo General")["Gasto"].sum().sort_values(ascending=True)
    total_gasto = por_tipo.sum()
    porcentajes = (por_tipo / total_gasto * 100).round(1) if total_gasto != 0 else por_tipo * 0

    csv_path = out_dir / "gasto_por_tipo_general.csv"
    por_tipo.to_csv(csv_path, header=["Gasto"])
    print(f"✅ Guardado: {csv_path.name}")

    palette = pq.Set3 + pq.Set2 + pq.Set1 + pq.Pastel1 + pq.Pastel2
    colores = [palette[i % len(palette)] for i in range(len(por_tipo))]

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
                "Peso sobre total: %{customdata:.1f}%<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=f"Gasto por Tipo General  {str_fechas}",
        xaxis_title="€",
        yaxis_title="Tipo",
        margin=dict(l=100, r=40, t=60, b=40),
        height=max(350, len(por_tipo) * 30),
        showlegend=False,
    )
    utils.chart_guardar(fig, out_path=out_dir, nombre="04_gasto_por_tipo_general")


def vista_top_gastos(df: pd.DataFrame, out_dir: Path, str_fechas: str, top_n: int = 10):
    out_dir.mkdir(parents=True, exist_ok=True)
    if "Concepto" not in df.columns or "Gasto" not in df.columns:
        raise ValueError("El DataFrame debe tener columnas 'Concepto' y 'Gasto'.")

    por_cat = df.groupby("Concepto")["Gasto"].sum().sort_values(ascending=False).head(top_n)
    total_top = por_cat.sum()
    porcentajes = (por_cat / total_top * 100).round(1) if total_top != 0 else por_cat * 0

    csv_path = out_dir / "top_gastos.csv"
    por_cat.to_csv(csv_path, header=["Gasto"])
    print(f"✅ Guardado: {csv_path.name}")

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
                "Peso en Top: %{customdata:.1f}%<extra></extra>"
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
    utils.chart_guardar(fig, out_path=out_dir, nombre="05_top_gastos")


def vista_por_semana(df: pd.DataFrame, out_dir: Path, str_fechas: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    if "Fecha" not in df.columns or "Gasto" not in df.columns:
        raise ValueError("El DataFrame debe tener columnas 'Fecha' y 'Gasto'.")

    df = df.copy()
    df["Fecha"] = pd.to_datetime(df["Fecha"]).dt.normalize()
    df["Semana_Lunes"] = df["Fecha"] - pd.to_timedelta(df["Fecha"].dt.weekday, unit="D")

    por_semana = df.groupby("Semana_Lunes")["Gasto"].sum().sort_index()
    csv_path = out_dir / "gasto_por_semana.csv"
    por_semana.rename("Gasto").to_csv(csv_path, header=True)
    print(f"✅ Guardado: {csv_path.name}")

    palette = pq.Set3 + pq.Set2 + pq.Set1 + pq.Pastel1 + pq.Pastel2
    color = palette[0]
    x_vals = por_semana.index
    y_vals = por_semana.values

    fig = go.Figure(
        go.Bar(
            x=x_vals,
            y=y_vals,
            marker_color=color,
            text=[f"{v:,.0f} €" for v in y_vals],
            textposition="outside",
            hovertemplate=(
                "Semana que empieza: %{x|%Y-%m-%d}<br>" "Gasto total: %{y:,.2f} €<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=f"Gasto por semana (Lun→Dom)  {str_fechas}",
        xaxis_title="Semana (lunes)",
        yaxis_title="€",
        margin=dict(l=60, r=40, t=60, b=60),
        height=400,
        showlegend=False,
    )
    fig.update_xaxes(tickangle=-45)
    utils.chart_guardar(fig, out_path=out_dir, nombre="06_gasto_por_semana")


def vista_media_por_dia_semana(
    df: pd.DataFrame,
    out_dir: Path,
    str_fechas: str,
    incluir_dias_sin_gasto: bool = True,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    if "Gasto" not in df.columns:
        raise ValueError("El DataFrame debe tener columna 'Gasto'.")

    date_col = (
        "F.Valor" if "F.Valor" in df.columns else ("Fecha" if "Fecha" in df.columns else None)
    )
    if date_col is None:
        raise ValueError("Se requiere 'F.Valor' o 'Fecha' en el DataFrame.")

    dfx = df.copy()
    dfx[date_col] = pd.to_datetime(dfx[date_col])
    dfx["Fecha"] = dfx[date_col].dt.normalize()
    dfx["Gasto"] = pd.to_numeric(dfx["Gasto"], errors="coerce").fillna(0.0)

    gasto_diario = dfx.groupby("Fecha")["Gasto"].sum()

    if incluir_dias_sin_gasto and not gasto_diario.empty:
        all_days = pd.date_range(gasto_diario.index.min(), gasto_diario.index.max(), freq="D")
        gasto_diario = gasto_diario.reindex(all_days, fill_value=0.0)

    di = gasto_diario.to_frame("Gasto")
    di["DiaSemanaNum"] = di.index.weekday
    agg = di.groupby("DiaSemanaNum")["Gasto"].agg(media="mean", total="sum", n_dias="size")

    nombres = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    agg = agg.reindex(range(7))
    agg.insert(0, "DiaSemana", nombres)

    csv_path = out_dir / "media_y_total_gasto_por_dia_semana.csv"
    agg.to_csv(csv_path, index=False)
    print(f"✅ Guardado: {csv_path.name}")

    palette = pq.Set3 + pq.Set2 + pq.Set1 + pq.Pastel1 + pq.Pastel2
    color_bar = palette[0]
    color_line = "#E74C3C"

    total_max = float(agg["total"].max() or 0)
    media_max = float(agg["media"].max() or 0)

    fig = go.Figure()
    fig.add_bar(
        x=agg["DiaSemana"],
        y=agg["total"],
        name="Total",
        marker_color=color_bar,
        text=[f"{v:,.0f} €" for v in agg["total"].values],
        textposition="outside",
        cliponaxis=False,
        customdata=np.column_stack([agg["media"].values, agg["n_dias"].values]),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Total: %{y:,.2f} €<br>"
            "Media diaria: %{customdata[0]:,.2f} €<br>"
            "Nº de días: %{customdata[1]:,d}<extra></extra>"
        ),
    )
    fig.add_scatter(
        x=agg["DiaSemana"],
        y=agg["media"],
        name="Media diaria",
        mode="lines+markers+text",
        line=dict(color=color_line, width=3),
        text=[f"{v:,.0f} €" for v in agg["media"].values],
        textposition="bottom center",
        textfont=dict(size=11),
        cliponaxis=False,
        hovertemplate="<b>%{x}</b><br>Media diaria: %{y:,.2f} €<extra></extra>",
        yaxis="y2",
    )

    fig.update_layout(
        title=f"Gasto por día de la semana: Total (barras) y Media (línea)  {str_fechas}",
        xaxis_title="Día de la semana",
        yaxis=dict(title="€ (total)", range=[0, total_max * 1.18] if total_max > 0 else [0, 1]),
        yaxis2=dict(
            title="€ (media diaria)",
            overlaying="y",
            side="right",
            showgrid=False,
            range=([0, media_max * 1.8] if media_max > 0 else [0, 1]),
        ),
        margin=dict(l=60, r=70, t=60, b=60),
        legend_title_text="Métrica",
        hovermode="x unified",
        bargap=0.25,
    )
    utils.chart_guardar(fig, out_path=out_dir, nombre="07_media_y_total_gasto_por_dia_semana")


def _etiqueta_gasto(row: pd.Series) -> str:
    fecha = row["Fecha"].strftime("%d/%m")
    base = row["Concepto"] if row.get("Concepto", "") else row.get("Categoría", "")
    txt = f"{fecha} · {base}".strip(" ·")
    return (txt[:40] + "…") if len(txt) > 41 else txt


def vista_barras_por_tipo_detalle(df: pd.DataFrame, out_dir: Path, str_fechas: str, min_barras=1):
    carpeta = out_dir / "por_tipo"
    carpeta.mkdir(parents=True, exist_ok=True)

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
        dft = dft[dft["Gasto"] > 0]
        if dft.empty or len(dft) < min_barras:
            continue

        dft["etiqueta"] = dft.apply(_etiqueta_gasto, axis=1)
        dft = dft.sort_values("Gasto", ascending=False)

        color = palette[idx % len(palette)]
        width = max(900, min(1800, 22 * len(dft) + 600))

        fig = go.Figure(
            go.Bar(
                x=dft["etiqueta"],
                y=dft["Gasto"],
                marker_color=color,
                text=[utils.formato_euros(v) for v in dft["Gasto"]],
                textposition="outside",
                cliponaxis=False,
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
                    "Importe: %{y:,.2f} €<extra></extra>"
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
            width=width,
        )
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", str(tipo)).strip("_")
        utils.chart_guardar(fig, out_path=carpeta, nombre=f"tipo_{slug}")


def vista_histograma_gastos(df: pd.DataFrame, out_dir: Path, str_fechas: str, nbins: int = 20):
    if "Gasto" not in df.columns:
        raise ValueError("El DataFrame debe tener columna 'Gasto'.")
    gastos = pd.to_numeric(df["Gasto"], errors="coerce")
    gastos = gastos[gastos > 0].dropna()
    if gastos.empty:
        return

    mean_val = gastos.mean()
    median_val = gastos.median()
    counts, edges = np.histogram(gastos.values, bins=nbins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    widths = edges[1:] - edges[:-1]

    palette = pq.Set3 + pq.Set2 + pq.Set1 + pq.Pastel1 + pq.Pastel2
    bar_color = palette[2]

    fig = go.Figure(
        go.Bar(
            x=centers,
            y=counts,
            width=widths,
            marker_color=bar_color,
            customdata=np.column_stack([edges[:-1], edges[1:]]),
            hovertemplate=(
                "Rango: %{customdata[0]:,.2f}–%{customdata[1]:,.2f} €<br>"
                "Frecuencia: %{y:,d}<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=mean_val, line_dash="dash", line_width=2, line_color="#2E86C1")
    fig.add_vline(x=median_val, line_dash="dot", line_width=2, line_color="#E74C3C")
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
    utils.chart_guardar(fig, out_path=out_dir, nombre="09_histograma_gastos")


def vista_totales_por_mes(df: pd.DataFrame, out_dir: Path, str_fechas: str, top_k: int = 12):
    """
    Barras por mes (Ingreso/Gasto/Ahorro) lado a lado, apiladas por Tipo General.
    Colores consistentes por Tipo General a lo largo de todos los meses.
    Tooltip incluye el desglose de % de cada Tipo dentro del Tipo General.
    """
    # Requisitos mínimos (ahora también exige 'Tipo' para calcular el desglose)
    req = {"AñoMesStr", "Ingreso", "Gasto", "Ahorro", "Tipo General", "Tipo"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas: {missing}")

    dfx = df.copy()
    dfx["AñoMesStr"] = dfx["AñoMesStr"].astype(str)
    meses = sorted(dfx["AñoMesStr"].unique())

    # Conjunto estable de Tipos Generales y mapa de color consistente
    tipos_gen = [t for t in dfx["Tipo General"].dropna().astype(str).unique() if t.strip() != ""]
    tipos_gen = sorted(tipos_gen)  # orden estable para leyenda/colores

    palette = pq.Set3 + pq.Set2 + pq.Set1 + pq.Pastel1 + pq.Pastel2
    color_by_tipo_gen = {t: palette[i % len(palette)] for i, t in enumerate(tipos_gen)}

    fig = go.Figure()
    metricas = ["Ingreso", "Gasto", "Ahorro"]

    # ============================
    # Precalcular desglose % por (Mes, Tipo General) y por Métrica
    # ============================
    breakdown_per_metric = {}
    for met in metricas:
        # Suma por Mes, Tipo General y Tipo
        by_gt = dfx.groupby(["AñoMesStr", "Tipo General", "Tipo"])[met].sum().reset_index()
        br_map = {}
        for (mes, gen), sub in by_gt.groupby(["AñoMesStr", "Tipo General"]):
            total = float(sub[met].sum())
            if total <= 0:
                br_map[(mes, gen)] = "—"
                continue
            sub = sub[sub[met] > 0].sort_values(met, ascending=False)

            partes = [f"{row['Tipo']}: {row[met] / total * 100:.1f}%" for _, row in sub.iterrows()]
            br_map[(mes, gen)] = "<br>".join(partes) if partes else "—"
        breakdown_per_metric[met] = br_map

    # ============================
    # Construcción de trazas apiladas por Tipo General
    # ============================
    for met in metricas:
        metric_df = dfx.groupby(["AñoMesStr", "Tipo General"])[met].sum().reset_index()
        pivot = (
            metric_df.pivot(index="AñoMesStr", columns="Tipo General", values=met)
            .reindex(meses)
            .fillna(0.0)
        )

        for gen in tipos_gen:
            y_vals = (
                pivot[gen].values.astype(float)
                if gen in pivot.columns
                else np.zeros(len(meses), dtype=float)
            )
            if np.allclose(y_vals, 0.0):
                continue

            # Desglose por mes dentro de este Tipo General para esta métrica
            desgloses = [breakdown_per_metric[met].get((mes, gen), "—") for mes in meses]

            fig.add_bar(
                name=gen,
                x=meses,
                y=y_vals,
                marker_color=color_by_tipo_gen[gen],
                offsetgroup=met,  # barras Ingreso/Gasto/Ahorro lado a lado
                legendgroup=gen,  # misma entrada de leyenda para todas las métricas
                showlegend=(met == "Gasto"),
                customdata=np.column_stack(
                    [
                        np.array([met] * len(meses), dtype=object),  # 0: métrica
                        np.array([gen] * len(meses), dtype=object),  # 1: tipo general
                        np.array(desgloses, dtype=object),  # 2: desglose % por Tipo
                    ]
                ),
                hovertemplate=(
                    "Mes: %{x}<br>"
                    "Métrica: %{customdata[0]}<br>"
                    "Tipo general: %{customdata[1]}<br>"
                    "Importe: %{y:,.2f} €<br>"
                    "<br><b>Desglose por Tipo</b><br>%{customdata[2]}<extra></extra>"
                ),
            )

    # ============================
    # Línea de Ahorro Acumulado (Desahorro resta)
    # ============================

    dfx["Desahorro"] = np.where(dfx["Tipo General"] == "Desahorro", dfx["Importe"], 0.0)

    mensual_ahorro = dfx.groupby("AñoMesStr")["Ahorro"].sum().reindex(meses).fillna(0.0)
    mensual_desahorro = dfx.groupby("AñoMesStr")["Desahorro"].sum().reindex(meses).fillna(0.0)
    mensual_neto = (mensual_ahorro - mensual_desahorro).astype(float)
    acumulado = mensual_neto.cumsum()

    fig.add_scatter(
        name="Ahorro acumulado",
        x=meses,
        y=acumulado.values,
        mode="lines+markers",
        hovertemplate=(
            "Mes: %{x}<br>"
            "Ahorro mes: %{customdata[0]:,.2f} €<br>"
            "Desahorro mes: %{customdata[1]:,.2f} €<br>"
            "Neto mes: %{customdata[2]:,.2f} €<br>"
            "<b>Acumulado: %{y:,.2f} €</b><extra></extra>"
        ),
        customdata=np.column_stack(
            [mensual_ahorro.values, mensual_desahorro.values, mensual_neto.values]
        ),
    )

    # ============================
    # Línea de Balance
    # ============================
    mensual_ingreso = dfx.groupby("AñoMesStr")["Ingreso"].sum().reindex(meses).fillna(0.0)
    mensual_gasto = dfx.groupby("AñoMesStr")["Gasto"].sum().reindex(meses).fillna(0.0)
    mensual_balance = ((mensual_ingreso) - mensual_gasto).astype(float)

    fig.add_scatter(
        name="Balance Mensual",
        x=meses,
        y=mensual_balance.values,
        mode="lines+markers",
        hovertemplate=(
            "Mes: %{x}<br>"
            "Ingreso mes: %{customdata[0]:,.2f} €<br>"
            "Gasto mes: %{customdata[1]:,.2f} €<br>"
            "Neto mes: %{customdata[2]:,.2f} €<br>"
        ),
        customdata=np.column_stack(
            [mensual_ingreso.values, mensual_gasto.values, mensual_balance.values]
        ),
    )

    fig.update_layout(
        title=f"Totales por mes (apilado por Tipo General)  {str_fechas}",
        barmode="relative",
        bargap=0.20,
        bargroupgap=0.08,
        xaxis_title="Mes",
        yaxis=dict(
            title="€",
            tick0=0,
            dtick=2000,  # 👈 ticks cada 2000 €
            showgrid=True,
            gridcolor="LightGray",
        ),
        legend_title_text="Tipo General",
        margin=dict(l=60, r=40, t=70, b=60),
    )

    labels = [utils._mes_label(m) for m in meses]

    fig.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=meses,
        tickmode="array",
        tickvals=meses,
        ticktext=labels,
        tickangle=0,
    )

    utils.chart_guardar(fig, out_path=out_dir, nombre="00_totales_por_mes_tipo_general", show=True)


def vista_mes_por_tipo_general(df: pd.DataFrame, out_dir: Path, str_fechas: str):
    # Requisitos
    req = {"AñoMesStr", "Gasto", "Tipo", "Tipo General", "Concepto"}
    if not req.issubset(df.columns):
        raise ValueError(f"Faltan columnas: {req - set(df.columns)}")

    # Totales por mes y tipo general (lo que se grafica)
    grp = (
        df.groupby(["AñoMesStr", "Tipo General"])["Gasto"]
        .sum()
        .unstack("Tipo General")
        .fillna(0.0)
        .sort_index()
    )

    # Desglose por tipo, tipo general y concepto para el tooltip
    det = (
        df.assign(
            Tipo=df["Tipo"].fillna("(Sin tipo)"),
            Concepto=df["Concepto"].fillna("(Sin concepto)"),
        )
        .groupby(["AñoMesStr", "Tipo General", "Tipo", "Concepto"], dropna=False)["Gasto"]
        .sum()
        .reset_index()
    )

    TOP_N = 20

    # {(mes, tipo_general) -> "• Tipo | Tipo General | Concepto: 12,34 €<br>..."}
    breakdown_map = {}
    for (mes, tipo_general), sub in det.groupby(["AñoMesStr", "Tipo General"]):
        sub_sorted = sub.sort_values("Gasto", ascending=False)
        top = sub_sorted.head(TOP_N)
        otros_total = sub_sorted["Gasto"].iloc[TOP_N:].sum()

        lineas = [
            f"• {t} | {tg} | {c}: {v:,.2f} €"
            for t, tg, c, v in zip(top["Tipo"], top["Tipo General"], top["Concepto"], top["Gasto"])
        ]
        if abs(otros_total) > 1e-9:
            lineas.append(f"• Otros: {otros_total:,.2f} €")

        breakdown_map[(mes, tipo_general)] = "<br>".join(lineas) if lineas else "—"

    x = grp.index.tolist()
    fig = go.Figure()
    palette = pq.Set3 + pq.Set2 + pq.Set1 + pq.Pastel1 + pq.Pastel2

    # Un trace por Tipo General (barras apiladas)
    for i, tipo_general in enumerate(grp.columns):
        y_vals = grp[tipo_general].values.astype(float)
        text_vals = [breakdown_map.get((mes, tipo_general), "—") for mes in x]

        fig.add_bar(
            name=str(tipo_general),
            x=x,
            y=y_vals,
            marker_color=palette[i % len(palette)],
            text=text_vals,  # solo para tooltip
            textposition="none",
            hovertemplate=(
                "Mes: %{x}<br>"
                f"Tipo General: <b>{str(tipo_general)}</b><br>"
                "Total: %{y:,.2f} €"
                "<br><br><b>Desglose</b><br>"
                "Tipo | Tipo General | Concepto<br>"
                "%{text}"
                "<extra></extra>"
            ),
        )

    fig.update_layout(
        title=f"Gasto por mes y Tipo General (apilado)  {str_fechas}",
        barmode="stack",
        xaxis_title="Mes",
        yaxis_title="€",
        margin=dict(l=70, r=40, t=60, b=60),
        legend_title_text="Tipo General",
        hovermode="closest",
    )

    utils.chart_guardar(fig, out_path=out_dir, nombre="01_gasto_por_mes_y_tipo_general", show=True)


def vista_heatmap_dia_semana_mes(df: pd.DataFrame, out_dir: Path, str_fechas: str):
    if not {"Fecha", "Gasto", "AñoMesStr"}.issubset(df.columns):
        raise ValueError("Se requieren 'Fecha','Gasto','AñoMesStr'.")

    d = df.copy()
    d["Fecha"] = pd.to_datetime(d["Fecha"]).dt.normalize()
    daily = d.groupby(["AñoMesStr", "Fecha"])["Gasto"].sum().reset_index()

    # 0=Lun .. 6=Dom
    daily["dow"] = pd.to_datetime(daily["Fecha"]).dt.weekday

    # Media por día de semana y mes
    heat = daily.groupby(["AñoMesStr", "dow"])["Gasto"].mean().unstack("dow")

    # 🔧 Asegura las 7 columnas (0..6), aunque falten en los datos
    heat = heat.reindex(columns=range(7), fill_value=0.0)

    # Ordena por mes
    heat = heat.sort_index()

    # Nombres bonitos
    idx2name = {0: "Lun", 1: "Mar", 2: "Mié", 3: "Jue", 4: "Vie", 5: "Sáb", 6: "Dom"}
    heat.rename(columns=idx2name, inplace=True)
    x_labels = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]

    fig = go.Figure(
        data=go.Heatmap(
            z=heat[x_labels].values,  # usa el orden deseado
            x=x_labels,
            y=heat.index.tolist(),
            hovertemplate="Mes: %{y}<br>Día: %{x}<br>Media gasto: %{z:,.2f} €<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Heatmap gasto medio por día de semana y mes  {str_fechas}",
        xaxis_title="Día de semana",
        yaxis_title="Mes",
        margin=dict(l=80, r=40, t=60, b=60),
    )
    utils.chart_guardar(fig, out_path=out_dir, nombre="02_heatmap_dia_semana_vs_mes")


def vista_boxplot_importes_por_mes(df: pd.DataFrame, out_dir: Path, str_fechas: str):
    if not {"AñoMesStr", "Gasto"}.issubset(df.columns):
        raise ValueError("Se requieren 'AñoMesStr','Gasto'.")
    fig = go.Figure()
    for periodo, dmes in df.groupby("AñoMesStr"):
        vals = pd.to_numeric(dmes["Gasto"], errors="coerce")
        vals = vals[vals > 0].dropna()
        if vals.empty:
            continue
        fig.add_box(
            y=vals.values.astype(float),
            name=str(periodo),
            boxmean=True,
            hovertemplate="Mes: " + str(periodo) + "<br>€: %{y:,.2f}<extra></extra>",
        )
    fig.update_layout(
        title=f"Distribución de importes de gasto por mes  {str_fechas}",
        yaxis_title="€ por operación",
        margin=dict(l=60, r=40, t=60, b=60),
        showlegend=False,
    )
    utils.chart_guardar(fig, out_path=out_dir, nombre="03_boxplot_importes_por_mes")
