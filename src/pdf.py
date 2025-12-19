# src/pdf.py

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Optional

import pandas as pd
import pdfplumber


DATE_RE = re.compile(r"^(?P<d1>\d{2}/\d{2})\s+(?P<d2>\d{2}/\d{2})\s+(?P<rest>.*)$")
EMISION_RE = re.compile(r"Fechadeemisi[oó]n:\s*(\d{2})/(\d{2})/(\d{4})", re.IGNORECASE)
EXTRACTO_MES_RE = re.compile(r"EXTRACTO\s*DE\s*([A-ZÁÉÍÓÚÑ]+)\s*(\d{4})", re.IGNORECASE)
MESES_ES = {
    "ENERO": 1,
    "FEBRERO": 2,
    "MARZO": 3,
    "ABRIL": 4,
    "MAYO": 5,
    "JUNIO": 6,
    "JULIO": 7,
    "AGOSTO": 8,
    "SEPTIEMBRE": 9,
    "OCTUBRE": 10,
    "NOVIEMBRE": 11,
    "DICIEMBRE": 12,
}


def parse_es_decimal(s: str) -> Decimal:
    """
    Spanish money format examples:
    3.518,03
    1.827,34
    -44,17
    """
    t = s.strip().replace(".", "").replace(",", ".")
    try:
        return Decimal(t)
    except InvalidOperation:
        raise ValueError(f"Cannot parse amount: {s!r}")


def parse_amount_token(token: str) -> Decimal:
    """
    Accepts either negative or positive tokens.
    Uses Decimal negate without writing the hyphen operator in code.
    """
    raw = token.strip()
    is_negative = raw.startswith("-")
    if is_negative:
        raw = raw[1:].strip()
    value = parse_es_decimal(raw)
    return value.copy_negate() if is_negative else value


def parse_bbva_monthly_pdf(pdf_path: str | Path) -> pd.DataFrame:
    rows: list[dict] = []
    pdf_path = Path(pdf_path)

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

            pending: Optional[dict] = None

            for ln in lines:
                m = DATE_RE.match(ln)
                if m:
                    if pending:
                        rows.append(pending)
                        pending = None

                    rest = m.group("rest")

                    tokens = rest.split()
                    amount_token = None
                    balance_token = None

                    if len(tokens) >= 2:
                        last = tokens[-1]
                        prev = tokens[-2]
                        if re.fullmatch(r"-?[\d.]+,\d{2}", last) and re.fullmatch(
                            r"-?[\d.]+,\d{2}", prev
                        ):
                            balance_token = last
                            amount_token = prev
                            concept = " ".join(tokens[:-2]).strip()
                        else:
                            concept = rest.strip()
                    else:
                        concept = rest.strip()

                    pending = {
                        "page": page_index,
                        "f_oper": m.group("d1"),
                        "f_valor": m.group("d2"),
                        "concepto": concept,
                        "importe": parse_amount_token(amount_token) if amount_token else None,
                        "saldo": parse_amount_token(balance_token) if balance_token else None,
                        "raw_line": ln,
                    }
                else:
                    if pending:
                        pending["concepto"] = (pending["concepto"] + " " + ln).strip()

            if pending:
                rows.append(pending)

    df = pd.DataFrame(rows)

    df = df[df["importe"].notna()]

    return df.reset_index(drop=True)


def load_statement(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".xlsx":
        return pd.read_excel(path)
    if suffix == ".pdf":
        return parse_bbva_monthly_pdf(path)

    raise ValueError(f"Unsupported file type: {suffix}")


def extraer_anyo_pdf(pdf_path: str | Path) -> int:
    pdf_path = Path(pdf_path)
    with pdfplumber.open(str(pdf_path)) as pdf:
        text = pdf.pages[0].extract_text() or ""

    m = EMISION_RE.search(text)
    if m:
        return int(m.group(3))

    m = EXTRACTO_MES_RE.search(text.replace("\n", " "))
    if m:
        return int(m.group(2))

    raise ValueError(f"No se pudo extraer el año del PDF: {pdf_path}")
