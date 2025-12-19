# /src/utils.py

import sys
import re
from pathlib import Path
import plotly.graph_objects as go


def base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent.resolve()
    # En desarrollo: .../PROJECT_ROOT/src/utils.py  -> PROJECT_ROOT
    return Path(__file__).resolve().parents[1]


def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (base_dir() / p)


def asegurar_dir(out_root: Path) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    return out_root


def formato_euros(valor: float) -> str:
    return f"{valor:,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")


def chart_guardar(fig: go.Figure, out_path: Path, nombre: str, show=False):
    out_path.mkdir(parents=True, exist_ok=True)
    html_path = out_path / f"{nombre}.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    if show:
        fig.show()
    print(f"✅ Guardado: {html_path}")


def _mes_label(m: str) -> str:
    parts = re.findall(r"\d+", str(m))
    year = parts[0] if len(parts) > 0 else ""
    month = parts[1] if len(parts) > 1 else ""
    month_map = {
        "01": "Jan",
        "02": "Feb",
        "03": "Mar",
        "04": "Apr",
        "05": "May",
        "06": "Jun",
        "07": "Jul",
        "08": "Aug",
        "09": "Sep",
        "10": "Oct",
        "11": "Nov",
        "12": "Dec",
    }
    return f"{month_map.get(month, month)} {year}".strip()
