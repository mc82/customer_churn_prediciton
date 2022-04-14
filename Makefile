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
	python3 -m src.churn_script_logging_and_tests
	
lint:
	find ./src ./test -type f -name "*.py" | xargs pylint

autopep:
	find ./src ./test -type f -name "*.py" | xargs autopep8 --in-place --aggressive --aggressive 

create_zip:
	zip -X -x 'env/*' -x 'tmp/*' -x 'mypy_cache/*'-x '*.mypy_cache*' -x '-x 'churn_prediction.zip'.git/*' -r churn_prediction.zip .

# show_zip:
# 	rm -rf tmp 
# 	mkdir tmp 
# 	cp churn_prediction.zip tmp/ \
# 	cd tmp/ \
# 	# unzip churn_prediction.zip