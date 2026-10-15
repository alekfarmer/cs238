# Every command in this project runs through this Makefile, inside Docker.
# `make help` lists the targets; `make verify` is the gate.

SHELL := /bin/bash
.DEFAULT_GOAL := help

IMAGE  := cs238-project1
BRANCH := project1-structure-learning
REPO   := alekfarmer/cs238

# Run as the host user so files written to the mounted repo are owned by you.
DOCKER_RUN := docker run --rm --user $$(id -u):$$(id -g) -v "$(CURDIR):/work" -w /work

P1  := project1
OUT := $(P1)/output

# Search settings per dataset (see `python project1.py --help`).
SEARCH_small  := --max-parents 4 --k2-orderings 200 --perturbations 200 --seed 0
SEARCH_medium := --max-parents 4 --k2-orderings 100 --perturbations 100 --seed 0
SEARCH_large  := --max-parents 4 --k2-orderings 30  --perturbations 40  --seed 0

.PHONY: help build build-tools lint format test verify \
        learn-small learn-medium learn-large learn-all clean push pr

help: ## List targets
	@grep -E '^[a-zA-Z_%-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

build: ## Build the dev image (runtime + test tools)
	docker build -f docker/Dockerfile --target dev -t $(IMAGE):dev .

build-tools: ## Build the GitHub CLI image used by `make pr`
	docker build -f docker/Dockerfile --target tools -t $(IMAGE):tools .

lint: build ## Ruff lint + format check, mypy strict type check
	$(DOCKER_RUN) $(IMAGE):dev sh -c "ruff check . && ruff format --check . && mypy"

format: build ## Apply ruff formatting and safe fixes
	$(DOCKER_RUN) $(IMAGE):dev sh -c "ruff check --fix . && ruff format ."

test: build ## Run the test suite
	$(DOCKER_RUN) $(IMAGE):dev pytest -q

verify: lint test ## The gate: lint, type check and tests must all pass

learn-%: build ## Learn a structure for one dataset: learn-small | learn-medium | learn-large
	@test -n '$(SEARCH_$*)' || { echo "unknown dataset '$*' (use small, medium or large)"; exit 2; }
	$(DOCKER_RUN) $(IMAGE):dev python $(P1)/project1.py \
		$(P1)/data/$*.csv $(OUT)/$*.gph --summary $(OUT)/$*.json $(SEARCH_$*)
	$(DOCKER_RUN) $(IMAGE):dev python $(P1)/plot_graph.py \
		$(P1)/data/$*.csv $(OUT)/$*.gph $(OUT)/$*.png --title "$*.csv"

learn-all: learn-small learn-medium learn-large ## Learn all three structures

clean: ## Remove caches (keeps learned outputs)
	rm -rf .pytest_cache .mypy_cache .ruff_cache $(P1)/__pycache__ $(P1)/tests/__pycache__

push: ## Push the feature branch to your fork (origin)
	git push -u origin $(BRANCH)

pr: build-tools ## Open a PR from the feature branch into main of your fork (needs GH_TOKEN in .env)
	@test -f .env || { echo ".env missing: copy .env.example to .env and set GH_TOKEN"; exit 2; }
	@test -f .pr-body.md || { echo ".pr-body.md missing"; exit 2; }
	docker run --rm --env-file .env -v "$(CURDIR):/work" -w /work $(IMAGE):tools \
		gh pr create --repo $(REPO) --base main --head $(BRANCH) \
		--title "Project 1: Bayesian network structure learning" --body-file .pr-body.md
