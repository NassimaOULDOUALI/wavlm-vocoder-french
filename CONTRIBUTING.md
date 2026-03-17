# Contributing to WavLM Vocoder for French

Thank you for your interest in contributing to this project.
This repository is a research-oriented codebase for French neural vocoding with WavLM-based conditioning. The goal of this guide is to make contributions easier, cleaner, and more reproducible.

## Table of contents

- [Code of conduct](#code-of-conduct)
- [How to contribute](#how-to-contribute)
  - [Report a bug](#report-a-bug)
  - [Suggest an improvement](#suggest-an-improvement)
  - [Submit a pull request](#submit-a-pull-request)
- [Development setup](#development-setup)
- [Coding standards](#coding-standards)
- [Tests](#tests)
- [Documentation](#documentation)
- [Research and reproducibility guidelines](#research-and-reproducibility-guidelines)
- [Review process](#review-process)
- [What should not be committed](#what-should-not-be-committed)
- [License](#license)

## Code of conduct

This project follows the principles of the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
Please be respectful, constructive, and professional in all interactions.

## How to contribute

### Report a bug

If you find a bug, please open an issue and include the following information:

- A short and explicit title, for example: `[Bug] Training crashes when loading manifests`
- The expected behavior
- The observed behavior
- Steps to reproduce the issue
- Your environment:
  - Operating system
  - Python version
  - PyTorch version
  - CUDA version, if relevant
- The full error message or traceback
- Any configuration file or command used to reproduce the problem

Before opening a new issue, please check whether the bug has already been reported.

### Suggest an improvement

If you want to propose an enhancement, a refactor, or a research-oriented extension, open an issue with a title such as:

`[Feature] Add evaluation script for spectral metrics`

Please explain:

- The problem or limitation
- The proposed solution
- Why the change is useful
- Any relevant references, papers, or implementation notes

### Submit a pull request

To submit a contribution:

1. Fork the repository.
2. Create a dedicated branch from `main`.
3. Keep your change focused on one topic.
4. Add or update tests when relevant.
5. Update documentation if your change affects usage, training, evaluation, or configuration.
6. Open a pull request with a clear description.

Example:

```bash
git checkout -b feature/improve-eval-pipeline
```

Recommended branch prefixes:

- `feature/` for new features
- `fix/` for bug fixes
- `refactor/` for code cleanup
- `docs/` for documentation-only changes
- `test/` for test-related work

A good pull request should:

- Be small enough to review easily
- Explain the motivation of the change
- Reference the related issue when applicable
- Avoid unrelated formatting or large incidental changes

## Development setup

Clone your fork and install the project in a virtual environment.

```bash
git clone https://github.com/NassimaOULDOUALI/wavlm-vocoder-french.git
cd wavlm-vocoder-french

python -m venv .venv
source .venv/bin/activate
# On Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

If you use additional development tools locally, you may also install:

```bash
pip install pytest pytest-cov black isort flake8 mypy
```

## Coding standards

Please keep the code simple, readable, and consistent.

### General guidelines

- Prefer explicit and maintainable code over clever shortcuts.
- Use descriptive variable and function names.
- Keep functions reasonably small and focused.
- Add comments only when they clarify non-obvious reasoning.
- Use docstrings for public classes and functions.
- Avoid hard-coded local paths, machine-specific settings, and private infrastructure details.

### Formatting

Use Black for formatting and isort for import ordering.

```bash
black src tests scripts
isort src tests scripts
```

### Linting

```bash
flake8 src tests scripts --max-line-length=120
```

### Type hints

Type hints are encouraged, especially for public functions, dataset interfaces, model utilities, and trainer code.

```bash
mypy src --ignore-missing-imports
```

## Tests

All contributions should be validated before submission.

Run the test suite with:

```bash
pytest tests -v
```

If relevant, also run coverage:

```bash
pytest tests --cov=src --cov-report=term-missing
```

When adding a new feature:

- Add a unit test when possible
- Add an integration test if the feature affects training, evaluation, or inference behavior
- Make sure the change does not silently break existing scripts or configs

If your change is not easily testable with an automated test, explain in the pull request how it was validated.

## Documentation

Please update documentation whenever your change affects how the repository is used.

This may include:

- `README.md`
- files in `docs/`
- configuration examples in `configs/`
- command examples in training or evaluation instructions

Docstrings should follow a clear and consistent style. A Google-style docstring is recommended.

Example:

```python
def load_manifest(path: str) -> list[str]:
    """Load a text manifest file.

    Args:
        path: Path to the manifest file.

    Returns:
        A list of non-empty lines.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
```

## Research and reproducibility guidelines

Because this is a research repository, reproducibility matters.

Please follow these rules whenever possible:

- Do not hard-code absolute paths tied to a private machine or cluster.
- Keep experiment settings configurable through YAML or command-line arguments.
- Document important assumptions in the relevant config or script.
- When changing a model, loss, dataset pipeline, or training loop, explain the expected impact.
- When reporting results in a pull request, specify:
  - dataset or subset used
  - checkpoint or training stage
  - metrics computed
  - command or config used

If you introduce a new experiment setting, please prefer a dedicated config file rather than modifying a shared base config in a way that breaks existing workflows.

## Review process

After you open a pull request:

1. A maintainer will review the code.
2. You may be asked to revise parts of the implementation.
3. The pull request can be merged once the requested changes are addressed and the contribution is considered consistent with the repository.

Review will typically focus on:

- Correctness
- Clarity and maintainability
- Reproducibility
- Backward compatibility with existing scripts and configs
- Documentation quality
- Test coverage when relevant

## What should not be committed

Please do not commit the following:

- Large datasets
- Audio corpora or private recordings
- Model checkpoints unless explicitly intended for release
- Temporary files
- Notebook outputs
- Secrets, tokens, or credentials
- Absolute paths to private systems
- Machine-specific cache files

If needed, update `.gitignore` as part of your pull request.

## License

By contributing to this repository, you agree that your contributions will be distributed under the same license as the project.

Thank you again for contributing.
