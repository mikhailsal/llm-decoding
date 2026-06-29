# Convenience targets. Edit on the client, run on `dsbx-host`.
DSBX_HOST ?= dsbx-host
DSBX_DEST ?= llm-decoding
REMOTE = ssh $(DSBX_HOST) 'cd $(DSBX_DEST) && source .venv/bin/activate &&

.PHONY: help sync doctor probe doctor-local probe-local serve-py serve-hf fmt \
        web-dev web-build web-prod web-test test coverage lint quality-check

help:
	@echo "Targets:"
	@echo "  sync          rsync source to dsbx-host"
	@echo "  doctor        sync + run 'dsbx doctor' on dsbx-host"
	@echo "  probe         sync + run 'dsbx probe' on dsbx-host"
	@echo "  doctor-local  run 'dsbx doctor' on this machine (probes remotes too)"
	@echo "  probe-local   run 'dsbx probe' on this machine"
	@echo "  serve-py      sync + start 'dsbx serve --backend llamacpp-py' on dsbx-host (port 8000)"
	@echo "  serve-hf      sync + start 'dsbx serve --backend hf' on dsbx-host (port 8001)"
	@echo ""
	@echo "Quality:"
	@echo "  test          run the pytest suite"
	@echo "  coverage      run pytest with coverage (term + htmlcov/ report)"
	@echo "  lint          ruff check + ruff format --check"
	@echo "  quality-check lint + code limits + mypy (informational) + tests"
	@echo ""
	@echo "Web UI (middleware on this machine + SvelteKit frontend):"
	@echo "  web-dev       run middleware on :8765 with frontend dev server on :5173"
	@echo "  web-build     build the SvelteKit bundle into frontend/build/"
	@echo "  web-prod      run middleware on :8765 serving pre-built frontend from build/"
	@echo "  web-test      run pytest -k web and pnpm test (frontend unit tests)"

sync:
	scripts/sync_to_host.sh $(DSBX_DEST)

doctor: sync
	$(REMOTE) dsbx doctor'

probe: sync
	$(REMOTE) dsbx probe'

doctor-local:
	python -m decoding_sandbox.cli doctor

probe-local:
	python -m decoding_sandbox.cli probe

# `make serve-py` / `make serve-hf` are convenience wrappers that ssh into
# dsbx-host and launch a long-lived `dsbx serve`. Keep them in separate ports so
# both backends can run simultaneously.
serve-py: sync
	$(REMOTE) bash scripts/run_dsbx_server_host.sh llamacpp-py 8000'

serve-hf: sync
	$(REMOTE) bash scripts/run_dsbx_server_host.sh hf 8001'

fmt:
	ruff check --fix . || true
	ruff format . || true

# Quality gate (run locally before pushing; mirrors CI minus the matrix).
lint:
	ruff check .
	ruff format --check .

test:
	pytest -q

coverage:
	pytest --cov=decoding_sandbox --cov-report=term-missing --cov-report=html

quality-check: lint
	python scripts/check_code_limits.py
	mypy decoding_sandbox || true
	pytest -q

# Web UI. The middleware reads its bearer token from $DSBX_WEB_TOKEN; export
# one before invoking these targets, or put it in [web].api_token of
# config.toml (gitignored).
web-build:
	cd frontend && pnpm install && pnpm build

web-dev:
	@bash -c 'set -eo pipefail; \
	  test -d frontend/node_modules || (cd frontend && pnpm install); \
	  cd frontend && pnpm dev --host 127.0.0.1 --port 5173 & \
	  trap "kill $$!" INT TERM EXIT; \
	  cd .. ; \
	  dsbx web --host 127.0.0.1 --port 8765'

web-prod:
	@bash -c 'set -eo pipefail; \
	  test -d frontend/build || (cd frontend && pnpm install && pnpm build); \
	  dsbx web --host 127.0.0.1 --port 8765 --frontend-dist frontend/build'

web-test:
	pytest tests -k web
	cd frontend && pnpm test --run
