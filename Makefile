all: run_local

run_local:
	@echo "Running local training and inference..."
	./scripts/run_local.sh

run_kaggle:
	@echo "Running training and inference on Kaggle..."
	./scripts/run_kaggle.sh

download_model:
	@echo "Downloading the base model..."
	./scripts/download_model.sh

clean:
	@echo "Cleaning up temporary files..."
	rm -rf __pycache__/
	rm -rf *.pyc
	rm -rf *.pkl

test:
	@echo "Running tests..."
	pytest tests/