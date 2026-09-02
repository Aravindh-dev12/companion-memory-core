.PHONY: test smoke check

test:
	pytest -q

smoke:
	PYTHONPATH=src python eval/run.py --provider heuristic --preserve-turn-distance

check:
	python -m compileall -q src eval
	pytest -q
	PYTHONPATH=src python eval/run.py --provider heuristic --preserve-turn-distance
