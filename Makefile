.PHONY: setup-environment run lint

setup-environment:
	pip3 install --upgrade pip
	pip3 install virtualenv; \
	virtualenv env; \
	source env/bin/activate; \
	pip3 install -r requirements.txt; \
	deactivate 

run:
	python3 -m src.main

lint:
	find ./src ./test -type f -name "*.py" | xargs pylint