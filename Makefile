.PHONY: setup-environment run lint autopep test show_zip

setup-environment:
	pip3 install --upgrade pip
	pip3 install virtualenv; \
	virtualenv env; \
	source env/bin/activate; \
	pip3 install -r requirements.txt; \
	deactivate 

run:
	python3 -m src.churn_library

test:
	pytest src/churn_script_logging_and_tests.py --report-log=test_log.json

lint:
	find ./src ./test -type f -name "*.py" | xargs pylint

autopep:
	find ./src ./test -type f -name "*.py" | xargs autopep8 --in-place --aggressive --aggressive 

create_zip:
	zip -X -x '.vscode*' -x '*__pycache__*' -x 'env/*' -x 'tmp/*' -x 'mypy_cache/*'-x '*.mypy_cache*' -x 'churn_prediction.zip' -x '.git/*' -r churn_prediction.zip .

# show_zip:
# 	rm -rf tmp 
# 	mkdir tmp 
# 	cp churn_prediction.zip tmp/ \
# 	cd tmp/ \
# 	# unzip churn_prediction.zip