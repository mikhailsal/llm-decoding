# Convenience targets. Edit on the client, run on `dsbx-host`.
#
# `dsbx-host` is a cosmetic placeholder. Your real GPU box almost certainly
# has a different SSH alias / IP. Drop your local overrides into Makefile.local
# (gitignored) -- see Makefile.local.example for the recipe.
DSBX_HOST ?= dsbx-host
DSBX_DEST ?= dsbx
# Auto-heal the editable install on the remote. Background: the package was
# renamed `decoding_sandbox` -> `dsbx`; a stale .venv on the host still has
# an entry-point script importing the old name. We test the installed binary
# (NOT `import dsbx`, which would spuriously succeed via cwd on sys.path) and
# rerun `pip install -e .` if it fails. ~50 ms when healthy.
REMOTE = ssh $(DSBX_HOST) 'cd $(DSBX_DEST) && source .venv/bin/activate && \
           dsbx --version >/dev/null 2>&1 || pip install -e . >/dev/null &&

# Prefer the local editable install if one exists, so `make web-prod` works
# without the user remembering `source .venv/bin/activate`. Falls back to
# whatever `dsbx` is first on $$PATH.
DSBX_BIN := $(shell [ -x .venv/bin/dsbx ] && echo .venv/bin/dsbx || echo dsbx)

-include Makefile.local

.PHONY: help sync doctor probe doctor-local probe-local serve-py serve-hf fmt \
        web-dev web-build web-prod web-test test coverage lint quality-check \
        install-hooks _check_dsbx_host

help:
	@echo "Local override (gitignored):"
	@echo "  cp Makefile.local.example Makefile.local         # then edit DSBX_HOST"
	@echo "  cp config.local.example.toml >> config.toml      # for the [remote.*] block"
	@echo "  Current DSBX_HOST = $(DSBX_HOST)"
	@echo ""
	@echo "Remote (run on dsbx-host, code edited here):"
	@echo "  install-hooks Install git pre-commit hook"
	@echo "  sync          rsync source to \$$(DSBX_HOST)"
	@echo "  doctor        sync + run 'dsbx doctor' on \$$(DSBX_HOST)"
	@echo "  probe         sync + run 'dsbx probe' on \$$(DSBX_HOST)"
	@echo "  doctor-local  run 'dsbx doctor' on this machine (probes remotes too)"
	@echo "  probe-local   run 'dsbx probe' on this machine"
	@echo "  serve-py      sync + start 'dsbx serve --backend llamacpp-py' on \$$(DSBX_HOST) (:8000)"
	@echo "  serve-hf      sync + start 'dsbx serve --backend hf'         on \$$(DSBX_HOST) (:8001)"
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

# Guard the SSH-bound targets. The sanitization pass replaced the real host
# alias with the cosmetic `dsbx-host`; if the user hasn't supplied an override,
# fail loudly with a one-screen explanation instead of a cryptic rsync error.
_check_dsbx_host:
	@if [ "$(DSBX_HOST)" = "dsbx-host" ]; then \
	  echo "ERROR: DSBX_HOST is still the placeholder 'dsbx-host'."; \
	  echo ""; \
	  echo "  Pick one of:"; \
	  echo "    * cp Makefile.local.example Makefile.local  # then edit DSBX_HOST"; \
	  echo "    * make DSBX_HOST=<your-ssh-alias> $(MAKECMDGOALS)"; \
	  echo "    * export DSBX_HOST=<your-ssh-alias>"; \
	  echo ""; \
	  echo "  The Makefile.local file is gitignored so your real host name"; \
	  echo "  never enters the repo."; \
	  exit 1; \
	fi

sync: _check_dsbx_host
	DSBX_HOST=$(DSBX_HOST) scripts/sync_to_host.sh $(DSBX_DEST)

doctor: sync
	$(REMOTE) dsbx doctor'

probe: sync
	$(REMOTE) dsbx probe'

doctor-local:
	$(DSBX_BIN) doctor

probe-local:
	$(DSBX_BIN) probe

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
	pytest --cov=dsbx --cov-report=term-missing --cov-report=html

quality-check: lint
	python scripts/check_code_limits.py
	mypy dsbx || true
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
	  $(DSBX_BIN) web --host 127.0.0.1 --port 8765'

web-prod:
	@bash -c 'set -eo pipefail; \
	  test -d frontend/build || (cd frontend && pnpm install && pnpm build); \
	  $(DSBX_BIN) web --host 127.0.0.1 --port 8765 --frontend-dist frontend/build'

web-test:
	pytest tests -k web
	cd frontend && pnpm test --run

install-hooks:
	@echo "Installing pre-commit hook..."
	@mkdir -p .git/hooks
	@ln -sf ../../scripts/pre-commit .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit
	@echo "Pre-commit hook installed successfully (symlinked to scripts/pre-commit)."

