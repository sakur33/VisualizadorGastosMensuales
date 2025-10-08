# Makefile para visualizar_gastos.py (multimes) con datos en ./data/ y config en ./config/

PY := ./venv/bin/python
SCRIPT ?= src/main.py
OUT ?= ./salidas
DATA_DIR ?= ./data
CONFIG ?= ./config/categorias_de_gasto.json

GLOB ?=
INICIO ?=
FIN ?=

.PHONY: install run-test run-all clean

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install pandas numpy plotly openpyxl

# Usa el dataset de test en ./data/
run-test:
	@if [ ! -f "$(DATA_DIR)/test_movimientos.xlsx" ]; then \
		echo "❌ No existe $(DATA_DIR)/test_movimientos.xlsx"; exit 2; \
	fi
	$(PY) $(SCRIPT) --xls "$(DATA_DIR)/test_movimientos.xlsx" \
		--config "$(CONFIG)" \
		--salidas $(OUT)

# Ejecuta contra todo ./data/*.xlsx (puedes recortar con INICIO/FIN)
run-all:
	$(PY) $(SCRIPT) --glob "$(DATA_DIR)/*.xlsx" \
		--exclude "*/~$*.xlsx" \
		$(if $(INICIO),--inicio $(INICIO),) \
		$(if $(FIN),--fin $(FIN),) \
		--config "$(CONFIG)" \
		--salidas $(OUT)

# También puedes pasar un patrón diferente
run-glob:
	@if [ -z "$(GLOB)" ]; then \
		echo "Debes indicar GLOB, p.ej. make run-glob GLOB=\"./data/2025-*.xlsx\""; \
		exit 2; \
	fi
	$(PY) $(SCRIPT) --glob "$(GLOB)" \
		$(if $(INICIO),--inicio $(INICIO),) \
		$(if $(FIN),--fin $(FIN),) \
		--config "$(CONFIG)" \
		--salidas $(OUT)

clean:
	rm -rf $(OUT)
