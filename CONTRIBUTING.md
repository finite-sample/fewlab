# Contributing to fewlab

Thank you for your interest in contributing to fewlab! This guide will help you set up your development environment and understand our development workflow.

## Development Setup

### 1. Clone and Install

```bash
git clone https://github.com/finite-sample/fewlab.git
cd fewlab

# Install in development mode with all dev dependencies
pip install -e ".[dev]"
```

### 2. Set Up Pre-commit Hooks (Required)

We use pre-commit hooks to ensure code quality and prevent CI failures. **All contributors must install pre-commit hooks**.

```bash
# Install pre-commit hooks
pre-commit install

# Optional: Install for push events too (runs package build checks)
pre-commit install --hook-type pre-push
```

The pre-commit hooks will automatically:
- ✅ Run linting and formatting (Ruff)
- ✅ Execute the full test suite
- ✅ Check file formatting and syntax
- ✅ Validate package builds (on push)

**This mirrors our CI/CD pipeline locally, preventing failed builds.**

### 3. Verify Setup

Test that everything works:

```bash
# This should pass all checks
git add .
git commit -m "test: verify pre-commit setup"

# If you see hooks running and all pass, you're ready!
```

## Development Workflow

### Making Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our code standards:
   - Add type hints for new functions and variables
   - Use named constants instead of magic numbers (see `fewlab/constants.py`)
   - Follow existing code patterns and style

3. **Add tests** for new functionality:
   ```bash
   # Add tests to tests/ directory
   # Run tests locally
   pytest tests/ -v
   ```

4. **Commit changes**:
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   # Pre-commit hooks run automatically and must pass
   ```

### Code Quality Standards

Our pre-commit hooks enforce:

- **Linting**: Ruff checks for code quality issues
- **Formatting**: Ruff auto-formats code consistently
- **Tests**: All tests must pass before commit
- **Type Safety**: Type hints are required for new code
- **Constants**: No magic numbers (use `constants.py`)

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_api.py -v

# Run with coverage
pytest tests/ --cov=fewlab
```

### Manual Quality Checks

If you need to run checks manually:

```bash
# Linting
ruff check .

# Formatting
ruff format .

# Build package
python -m build

# All checks (same as pre-commit)
pre-commit run --all-files
```

## Pre-commit Hook Details

Our `.pre-commit-config.yaml` runs these checks:

### On Commit:
- **Ruff linting and formatting**: Ensures code quality and consistency
- **File checks**: Trailing whitespace, file endings, YAML/TOML syntax
- **Python checks**: AST validation, debug statement detection
- **Full test suite**: Ensures all tests pass (`pytest tests/ -v`)

### On Push:
- **Package build**: Verifies the package builds correctly (`python -m build`)

### Bypassing Hooks (Emergency Only)

```bash
# Skip hooks (use only for urgent fixes)
git commit --no-verify -m "emergency fix"

# Note: CI will still run all checks, so fix issues ASAP
```

## Troubleshooting

### Pre-commit Installation Issues

```bash
# Reinstall pre-commit
pip install --upgrade pre-commit
pre-commit clean
pre-commit install

# Update hooks to latest versions
pre-commit autoupdate
```

### Test Failures

```bash
# Check which tests are failing
pytest tests/ -v --tb=short

# Run linting to see specific issues
ruff check .

# Auto-fix formatting issues
ruff format .
```

### Hook Failures

```bash
# See what failed
pre-commit run --all-files

# Fix issues and try again
git add .
git commit -m "fix: resolve pre-commit issues"
```

## CI/CD Pipeline

Our GitHub Actions CI runs the same checks as pre-commit:

1. **Linting**: `ruff check .` and `ruff format --check .`
2. **Testing**: `pytest tests/ -v` on Python 3.9-3.12
3. **Building**: `python -m build` and installation test

**Pre-commit hooks ensure you never push code that will fail CI.**

## Need Help?

- Check existing issues and PRs for similar problems
- Run `pre-commit run --all-files` to see all current issues
- Open an issue if you're stuck on setup or contribution process

## Summary

1. Install dev dependencies: `pip install -e ".[dev]"`
2. Install pre-commit: `pre-commit install`
3. Make changes and commit (hooks run automatically)
4. All hooks must pass before your commit succeeds
5. Push and create PR (CI will run the same checks)

**The pre-commit hooks are your friend—they catch issues early and keep our codebase clean!**
