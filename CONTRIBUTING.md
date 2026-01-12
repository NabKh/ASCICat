# Contributing to ASCICat

Thank you for your interest in contributing to ASCICat! This document provides guidelines for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Contributions](#making-contributions)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)


## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- pip or conda for package management

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/nabkh/ASCICat.git
   cd ASCICat
   ```

3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/nabkh/ASCICat.git
   ```

## Development Setup

### Create a Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Or using conda
conda create -n ascicat python=3.10
conda activate ascicat
```

### Install Development Dependencies

```bash
pip install -e ".[dev]"
```

This installs ASCICat in development mode with all testing and documentation tools.

## Making Contributions

### Types of Contributions

We welcome:

1. **Bug Fixes**: Identify and fix bugs in the codebase
2. **New Features**: Add new functionality (please discuss first)
3. **Documentation**: Improve or extend documentation
4. **Tests**: Add or improve test coverage
5. **Examples**: Create new examples or tutorials
6. **Performance**: Optimize existing code

### Creating an Issue

Before starting work:

1. Check existing issues to avoid duplicates
2. Open a new issue describing:
   - The problem or feature request
   - Your proposed solution (if applicable)
   - Any relevant context or references

### Feature Branches

Create a feature branch for your work:

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

## Coding Standards

### Python Style

We follow PEP 8 with some modifications:

- Maximum line length: 100 characters
- Use double quotes for strings
- Use type hints for function signatures

### Docstrings

Use NumPy-style docstrings:

```python
def score_activity(delta_E: float, optimal_E: float, width: float) -> float:
    """
    Calculate activity score based on Sabatier principle.

    Parameters
    ----------
    delta_E : float
        Adsorption energy (eV)
    optimal_E : float
        Optimal binding energy (eV)
    width : float
        Scoring width parameter (eV)

    Returns
    -------
    float
        Activity score in [0, 1]

    Examples
    --------
    >>> score = score_activity(-0.27, -0.27, 0.15)
    >>> print(f"Score: {score:.3f}")
    Score: 1.000

    References
    ----------
    Norskov, J. K. et al. Nat. Chem. 1, 37 (2009)
    """
    pass
```

### Code Organization

- Keep modules focused on single responsibilities
- Use descriptive variable and function names
- Add comments for complex logic
- Include scientific references where applicable

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ascicat --cov-report=html

# Run specific test file
pytest tests/test_calculator.py

# Run specific test
pytest tests/test_calculator.py::test_her_calculation
```

### Writing Tests

- Place tests in the `tests/` directory
- Mirror the structure of `ascicat/` module
- Use descriptive test names
- Include edge cases and error conditions
- Add docstrings explaining what each test verifies

Example:

```python
def test_activity_score_at_optimum():
    """Test that activity score is 1.0 at optimal energy."""
    from ascicat.scoring import score_activity

    score = score_activity(-0.27, optimal_E=-0.27, width=0.15)
    assert score == 1.0, "Score at optimum should be exactly 1.0"
```

## Documentation

### Building Documentation

```bash
cd docs
make html
```

Documentation is built with Sphinx and hosted on ReadTheDocs.

### Adding Documentation

- Update docstrings in source code
- Add examples to `examples/` directory
- Update README.md for major features
- Add tutorials to `docs/tutorials/`

## Submitting Changes

### Before Submitting

1. Ensure all tests pass: `pytest`
2. Check code style: `flake8 ascicat/`
3. Format code: `black ascicat/`
4. Update documentation if needed
5. Add entry to CHANGELOG.md

### Pull Request Process

1. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Open a Pull Request on GitHub

3. Fill in the PR template with:
   - Description of changes
   - Related issue numbers
   - Testing performed
   - Screenshots (if applicable)

4. Wait for review and address feedback

### Review Process

- PRs require at least one approval
- CI tests must pass
- Documentation must be updated
- Code must follow style guidelines

## Scientific Contributions

When contributing scientific features:

1. **Cite Sources**: Include references in docstrings
2. **Validate**: Compare results with published data
3. **Document Assumptions**: Clearly state any approximations
4. **Provide Examples**: Show usage with realistic data

## Questions?

- Open an issue for general questions
- Email: n.khossossi@differ.nl 
- Check existing documentation

Thank you for contributing to ASCICat!
