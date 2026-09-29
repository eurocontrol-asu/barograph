.PHONY: install check test lint format security axm-audit axm-init quality ci docs-serve docs-build clean help

# Packages are DISCOVERED from [tool.uv.workspace] members = ["packages/*"]:
# adding a package needs no edit here. The test directory is derived from the
# package name (tests_<package_with_underscores>).
PACKAGES := $(notdir $(patsubst %/,%,$(dir $(wildcard packages/*/pyproject.toml))))

# 🚀 Workspace

install:  ## Install all packages and dependency groups
	uv sync --all-packages --all-groups

# 🧪 Tests

test:  ## Run every package's tests (each with its own configuration)
	@for pkg in $(PACKAGES); do \
		echo "\n📦 Testing $$pkg..."; \
		uv run --package $$pkg --directory packages/$$pkg pytest || exit 1; \
	done

# 🛡️ Quality gates

lint:  ## Ruff lint + format check, then mypy per package
	uv run ruff check .
	uv run ruff format --check .
	@for pkg in $(PACKAGES); do \
		tests="packages/$$pkg/tests_$$(echo $$pkg | tr - _)"; \
		[ -d "$$tests" ] || { echo "❌ $$pkg: $$tests not found"; exit 1; }; \
		uv run --package $$pkg mypy --config-file packages/$$pkg/pyproject.toml \
			packages/$$pkg/src "$$tests" || exit 1; \
	done

format:  ## Auto-format and auto-fix
	uv run ruff format .
	uv run ruff check --fix .

security:  ## Dependency vulnerability audit
	uv run pip-audit

check: lint test  ## Lint + type-check + tests

# 🏅 AXM quality gates (mirror .github/workflows/axm-quality.yml)

axm-audit:  ## axm audit of each package
	@for pkg in $(PACKAGES); do \
		echo "\n🔍 Auditing $$pkg..."; \
		uv run axm audit packages/$$pkg || exit 1; \
	done

axm-init:  ## axm init_check of the workspace root and each package
	uv run axm init_check .
	@for pkg in $(PACKAGES); do \
		echo "\n🏗️ Checking $$pkg..."; \
		uv run axm init_check packages/$$pkg || exit 1; \
	done

quality: axm-audit axm-init  ## Full AXM quality gate (pre-push)

ci: install check quality  ## Full CI pipeline

# 📚 Documentation

docs-serve:  ## Preview the documentation locally
	uv run mkdocs serve

docs-build:  ## Build the documentation site (strict)
	uv run mkdocs build --strict

# 🧹 Cleanup

clean:  ## Remove caches and build artefacts
	rm -rf .venv site
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
