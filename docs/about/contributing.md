# Contributing to ASCICat

We welcome contributions! This guide explains how to contribute to ASCICat.

## Ways to Contribute

1. **Bug Reports**: Found a bug? Open an issue
2. **Feature Requests**: Have an idea? Start a discussion
3. **Documentation**: Improve docs, fix typos
4. **Code**: Fix bugs, add features
5. **Examples**: Add tutorials, use cases

## Development Setup

```bash
# Clone repository
git clone https://github.com/NabKh/ASCICat.git
cd ASCICat

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/
```

## Code Standards

### Style

- Follow **PEP 8** guidelines
- Use **type hints** for function signatures
- Maximum line length: 88 characters (Black default)

### Docstrings

Use **NumPy-style docstrings**:

```python
def calculate_asci(
    activity_score: float,
    stability_score: float,
    cost_score: float,
    w_a: float = 0.33,
    w_s: float = 0.33,
    w_c: float = 0.34
) -> float:
    """
    Calculate combined ASCI score.

    Parameters
    ----------
    activity_score : float
        Activity score S_a in [0, 1]
    stability_score : float
        Stability score S_s in [0, 1]
    cost_score : float
        Cost score S_c in [0, 1]
    w_a : float, optional
        Activity weight (default: 0.33)
    w_s : float, optional
        Stability weight (default: 0.33)
    w_c : float, optional
        Cost weight (default: 0.34)

    Returns
    -------
    float
        Combined ASCI score in [0, 1]

    Examples
    --------
    >>> calculate_asci(0.9, 0.7, 0.8)
    0.80
    """
```

### Testing

- Write tests for new features
- Maintain test coverage > 80%
- Use pytest fixtures for shared setup

```python
# tests/test_scoring.py
import pytest
from ascicat.scoring import score_activity

def test_activity_at_optimal():
    """Activity score should be 1.0 at optimal energy."""
    score = score_activity(-0.27, optimal_E=-0.27, width=0.15)
    assert score == pytest.approx(1.0)

def test_activity_outside_window():
    """Activity score should be 0.0 outside window."""
    score = score_activity(-0.50, optimal_E=-0.27, width=0.15)
    assert score == 0.0
```

## Pull Request Process

1. **Fork** the repository
2. **Create branch**: `git checkout -b feature/my-feature`
3. **Make changes** with tests
4. **Run checks**:
   ```bash
   # Format code
   black ascicat/
   isort ascicat/

   # Type check
   mypy ascicat/

   # Tests
   pytest tests/ --cov=ascicat
   ```
5. **Commit**: Use descriptive messages
6. **Push**: `git push origin feature/my-feature`
7. **Open PR**: Against `master` branch

### Commit Messages

```
feat: Add OER reaction configuration
fix: Handle negative surface energies in stability scoring
docs: Add CO2RR tutorial
test: Add tests for sensitivity analysis
```

## Code Review

All PRs require review. Reviewers check:

- Code quality and style
- Test coverage
- Documentation updates
- Backwards compatibility

## Versioning

We use [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backwards compatible)
- **PATCH**: Bug fixes

## Questions?

- Open a [GitHub Discussion](https://github.com/NabKh/ASCICat/discussions)
- Email: [n.khossossi@differ.nl](mailto:n.khossossi@differ.nl)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
