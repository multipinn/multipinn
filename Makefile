.PHONY: format test clean install-pre-commit run-pre-commit


test:
	pytest --cov -vv -W ignore

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

install-pre-commit:
	pip install pre-commit
	pre-commit install

run-pre-commit:
	pre-commit run --all-files

all:
	make test
	make run-pre-commit

install.dev:
	@pip install -r requirements-dev.txt
	make install-pre-commit

install:
	@pip install -e .
