.PHONY: all sim iris

all: sim iris

sim:
	pipenv run python3 -m examples.sim

iris:
	pipenv run python3 -m examples.iris
