PYTHON ?= python
START_DATE ?= 2016-01-01
END_DATE ?= 2025-12-31

.PHONY: help install pipeline fetch-all fetch-water fetch-weather preprocess train metrics reliability hypotheses figures

help:
	@echo "Available targets:"
	@echo "  make install        Install Python dependencies"
	@echo "  make pipeline       Run preprocess + train with existing/default data"
	@echo "  make fetch-all      Fetch water/weather data, preprocess, and train"
	@echo "  make fetch-water    Re-fetch water/dam data, preprocess, and train"
	@echo "  make fetch-weather  Re-fetch weather data, preprocess, and train"
	@echo "  make preprocess     Build Final.csv/model input only"
	@echo "  make train          Train models from existing Final.csv"
	@echo "  make figures        Generate model reliability and hypothesis figures"
	@echo "  make reliability    Generate model reliability figures"
	@echo "  make hypotheses     Run hypothesis tests and figures"

install:
	$(PYTHON) -m pip install -r requirements.txt

pipeline:
	$(PYTHON) src/pipeline.py

fetch-all:
	$(PYTHON) src/pipeline.py --fetch all --start-date $(START_DATE) --end-date $(END_DATE)

fetch-water:
	$(PYTHON) src/pipeline.py --fetch water --start-date $(START_DATE) --end-date $(END_DATE)

fetch-weather:
	$(PYTHON) src/pipeline.py --fetch weather --start-date $(START_DATE) --end-date $(END_DATE)

preprocess:
	$(PYTHON) src/pipeline.py --skip-train

train:
	$(PYTHON) src/pipeline.py --skip-preprocess

metrics:
	$(PYTHON) src/plot_metrics.py

reliability:
	$(PYTHON) src/plot_reliability.py

hypotheses:
	$(PYTHON) src/run_hypothesis_tests.py

figures: metrics reliability hypotheses
