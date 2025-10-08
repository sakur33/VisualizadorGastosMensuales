# /src/utils.py

import sys
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


def chart_guardar(fig: go.Figure, out_path: Path, nombre: str):
    out_path.mkdir(parents=True, exist_ok=True)
    html_path = out_path / f"{nombre}.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    print(f"✅ Guardado: {html_path}")
