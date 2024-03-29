# Set the check directory
CHECK_DIR := ./

quality:
		black --check $(CHECK_DIR)
		isort --check-only  $(CHECK_DIR)
		flake8  $(CHECK_DIR)

format:
		black --fast $(CHECK_DIR)
		isort  $(CHECK_DIR)

clean:
		rm -rf .pytest_cache/
		rm -rf dcc.egg-info/
		rm -rf dist/
		rm -rf build/
		find . | grep -E '(\.mypy_cache|__pycache__|\.pyc|\.pyo$$)' | xargs rm -rf