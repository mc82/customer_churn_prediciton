.PHONY: setup-environment run

setup-environment:
	pip3 install --upgrade pip
	pip3 install virtualenv; \
	virtualenv env; \
	source env/bin/activate; \
	pip3 install -r requirements.txt; \
	deactivate 

run:
	python3 -m src.main