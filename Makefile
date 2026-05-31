.PHONY: lock install test lint typecheck

# ── Dependency management ──────────────────────────────────────────────────
# Compile requirements.in → requirements.txt with full version pins + SHA-256
# hashes.  Run this whenever requirements.in changes, then commit both files.
#
# Requires: pip install pip-tools
#
lock:
	pip-compile \
		--generate-hashes \
		--allow-unsafe \
		--output-file requirements.txt \
		requirements.in
	@echo "requirements.txt updated with hashes. Commit both files."

# Install from the locked + hashed file (verifies supply-chain integrity)
install:
	pip install --require-hashes --no-deps -r requirements.txt

# ── Quality gates ──────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

lint:
	ruff check app/ tests/

typecheck:
	mypy app/ --ignore-missing-imports
