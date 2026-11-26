.PHONY: install test lint run quick clean

install:
	pip install -r requirements.txt
	pip install -e .

lint:
	ruff check .

test:
	pytest -q

run:
	PYTHONPATH=src python -m energy_forecast.run --days 56

quick:
	PYTHONPATH=src python -m energy_forecast.run --days 14 --no-sarima

clean:
	rm -rf reports/*.csv reports/*.png .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
