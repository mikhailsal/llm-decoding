# Convenience targets. Edit on the client, run on `dsbx-host`.
DSBX_HOST ?= dsbx-host
DSBX_DEST ?= llm-decoding
REMOTE = ssh $(DSBX_HOST) 'cd $(DSBX_DEST) && source .venv/bin/activate &&

.PHONY: help sync doctor probe doctor-local probe-local fmt

help:
	@echo "Targets:"
	@echo "  sync          rsync source to dsbx-host"
	@echo "  doctor        sync + run 'dsbx doctor' on dsbx-host"
	@echo "  probe         sync + run 'dsbx probe' on dsbx-host"
	@echo "  doctor-local  run 'dsbx doctor' on this machine"
	@echo "  probe-local   run 'dsbx probe' on this machine"

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

fmt:
	ruff check --fix . || true
	ruff format . || true
