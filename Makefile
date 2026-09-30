.PHONY: install api ui test ingest
install:
	python -m pip install -r requirements.txt
api:
	uvicorn src.interfaces.api:app --reload --port 8000
ui:
	streamlit run src/interfaces/ui.py
test:
	pytest -q
ingest:
	python -m src.interfaces.cli ingest --recreate
