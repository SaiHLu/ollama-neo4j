setvenv:
	@python -m venv localenv

# activate:
# 	source localenv/bin/activate

deactivate:
	@deactivate

run1:
	@python '1. chains.py'

run2:
	@python '2. chatmodels.py'

run3:
	@python '3. in-memory-chat-history.py'

.PHONY: run setvenv deactivate