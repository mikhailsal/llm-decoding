# Convenience targets. Edit on the client, run on `dsbx-host`.
DSBX_HOST ?= dsbx-host
DSBX_DEST ?= llm-decoding
REMOTE = ssh $(DSBX_HOST) 'cd $(DSBX_DEST) && source .venv/bin/activate &&

.PHONY: help sync doctor probe doctor-local probe-local serve-py serve-hf fmt

help:
	@echo "Targets:"
	@echo "  sync          rsync source to dsbx-host"
	@echo "  doctor        sync + run 'dsbx doctor' on dsbx-host"
	@echo "  probe         sync + run 'dsbx probe' on dsbx-host"
	@echo "  doctor-local  run 'dsbx doctor' on this machine (probes remotes too)"
	@echo "  probe-local   run 'dsbx probe' on this machine"
	@echo "  serve-py      sync + start 'dsbx serve --backend llamacpp-py' on dsbx-host (port 8000)"
	@echo "  serve-hf      sync + start 'dsbx serve --backend hf' on dsbx-host (port 8001)"

sync:
	scripts/sync_to_wind.sh $(DSBX_DEST)

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
	$(REMOTE) bash scripts/run_dsbx_server_wind.sh llamacpp-py 8000'

serve-hf: sync
	$(REMOTE) bash scripts/run_dsbx_server_wind.sh hf 8001'

fmt:
	ruff check --fix . || true
	ruff format . || true
