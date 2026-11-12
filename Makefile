PYTHON ?= python
HOST ?= 127.0.0.1
PORT ?= 8000
export PYTHONPATH := src

.PHONY: install benchmark index index-sample evaluate test serve warmup

install:
	$(PYTHON) -m pip install -r requirements.txt

benchmark:  ## mide textos/s del encoder con 500 reseñas
	$(PYTHON) scripts/build_index.py --benchmark 500

index:  ## datos -> embeddings -> arquetipos (dataset completo, reanudable)
	$(PYTHON) scripts/build_index.py

index-sample:  ## prueba rápida en artifacts/sample/
	$(PYTHON) scripts/build_index.py --sample 2000

evaluate:  ## métricas en reports/metrics.json y figuras
	$(PYTHON) scripts/evaluate.py

test:
	$(PYTHON) -m pytest -q

warmup:  ## precalienta la caché del LLM con las consultas de ejemplo
	$(PYTHON) scripts/warmup.py

serve:  ## web en http://$(HOST):$(PORT)
	$(PYTHON) -m uvicorn restmex.app.main:app --app-dir src --host $(HOST) --port $(PORT)
