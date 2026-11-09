# Every command in this project runs through this Makefile, inside Docker.
# `make help` lists the targets; `make verify` is the gate.

SHELL := /bin/bash
.DEFAULT_GOAL := help

IMAGE  := cs238
BRANCH := project2-rl
PR_TITLE := Project 2: offline reinforcement learning
REPO   := alekfarmer/cs238

# Run as the host user so files written to the mounted repo are owned by you.
DOCKER_RUN := docker run --rm --user $$(id -u):$$(id -g) -v "$(CURDIR):/work" -w /work

P1  := project1
OUT := $(P1)/output

# Search settings per dataset (see `python project1.py --help`). Override on the
# command line to experiment, e.g. make learn-small SEARCH_small="--max-parents 2".
# The parent limit matters most; these values were chosen by trying 2..12.
SEARCH_small  := --max-parents 7 --k2-orderings 200 --perturbations 200 --seed 0
SEARCH_medium := --max-parents 12 --k2-orderings 200 --perturbations 300 --seed 0
SEARCH_large  := --max-parents 8 --k2-orderings 100 --perturbations 1000 --perturbation-moves 6 --seed 0

P2     := project2
P2_OUT := $(P2)/output

# Planning settings per dataset (see `python project2.py --help`). medium is
# undiscounted, so value iteration converges slowly; it gets a larger cap.
SOLVE_small  := --tol 1e-8 --values $(P2_OUT)/small.values
SOLVE_medium := --tol 1e-6 --max-iter 20000
SOLVE_large  := --tol 1e-8

.PHONY: help build build-tools lint format test verify \
        learn-small learn-medium learn-large learn-all \
        policy-small policy-medium policy-large policy-all plot-small plot-medium clean push pr

help: ## List targets
	@grep -E '^[a-zA-Z_ -]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-38s\033[0m %s\n", $$1, $$2}'

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

learn-small learn-medium learn-large: learn-%: build ## Learn one dataset's structure (.gph, .json, .png)
	$(DOCKER_RUN) $(IMAGE):dev python $(P1)/project1.py \
		$(P1)/data/$*.csv $(OUT)/$*.gph --summary $(OUT)/$*.json $(SEARCH_$*)
	$(DOCKER_RUN) $(IMAGE):dev python $(P1)/plot_graph.py \
		$(P1)/data/$*.csv $(OUT)/$*.gph $(OUT)/$*.png --title "$*.csv"

learn-all: learn-small learn-medium learn-large ## Learn all three structures

policy-small policy-medium policy-large: policy-%: build ## Learn one dataset's policy (.policy, .json)
	$(DOCKER_RUN) $(IMAGE):dev python $(P2)/project2.py $* \
		$(P2)/data/$*.csv $(P2_OUT)/$*.policy --summary $(P2_OUT)/$*.json \
		$(SOLVE_$*)

plot-small: policy-small ## Heatmap of small's values with policy arrows
	$(DOCKER_RUN) $(IMAGE):dev python $(P2)/plot_policy.py small \
		$(P2_OUT)/small.policy $(P2_OUT)/small.png --values $(P2_OUT)/small.values

plot-medium: policy-medium ## Map of medium's chosen action over (position, velocity)
	$(DOCKER_RUN) $(IMAGE):dev python $(P2)/plot_policy.py medium \
		$(P2_OUT)/medium.policy $(P2_OUT)/medium.png

policy-all: plot-small plot-medium policy-large ## Learn all three policies and draw the grid ones

clean: ## Remove caches (keeps learned outputs)
	rm -rf .pytest_cache .mypy_cache .ruff_cache $(P1)/__pycache__ $(P1)/tests/__pycache__ $(P2)/__pycache__ $(P2)/tests/__pycache__

push: ## Push the feature branch to your fork (origin)
	git push -u origin $(BRANCH)

pr: build-tools ## Open a PR from the feature branch into main of your fork (needs GH_TOKEN in .env)
	@test -f .env || { echo ".env missing: copy .env.example to .env and set GH_TOKEN"; exit 2; }
	@test -f .pr-body.md || { echo ".pr-body.md missing"; exit 2; }
	docker run --rm --env-file .env -v "$(CURDIR):/work" -w /work $(IMAGE):tools \
		gh pr create --repo $(REPO) --base main --head $(BRANCH) \
		--title "$(PR_TITLE)" --body-file .pr-body.md
