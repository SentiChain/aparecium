# Contributing to Aparecium

Thank you for your interest in contributing to Aparecium! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please report any unacceptable behavior to the project maintainers.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/aparecium.git
   cd aparecium
   ```
3. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. Set up your development environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev]"
   ```

## Development Guidelines

### Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) guidelines
- Use type hints for function parameters and return values
- Keep functions focused and single-purpose
- Write docstrings for all public functions and classes
- Use meaningful variable and function names

### Documentation

- Update documentation for any new features or changes
- Include docstring examples for complex functions
- Keep the README.md up to date with new features
- Add comments for complex algorithms or non-obvious code

### Testing

- Write unit tests for new features
- Ensure all tests pass before submitting a PR
- Maintain or improve test coverage
- Include integration tests for significant changes

### Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding or modifying tests
- `chore`: Maintenance tasks

Example:
```
feat(vectorizer): add support for batch processing

- Implement batch_encode method
- Add progress bar for large batches
- Update documentation with batch examples

Closes #123
```

## Pull Request Process

1. Update your fork with the latest changes:
   ```bash
   git remote add upstream https://github.com/SentiChain/aparecium.git
   git fetch upstream
   git rebase upstream/main
   ```

2. Push your changes to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

3. Create a Pull Request:
   - Use a clear, descriptive title
   - Reference any related issues
   - Include a detailed description of changes
   - Add screenshots or GIFs for UI changes
   - List any breaking changes

4. Respond to review comments and make requested changes

5. Once approved, squash and merge your PR

## Project Structure

```
aparecium/
├── aparecium/         # Main package code
├── examples/          # Example scripts
├── tests/            # Test suite
├── data/             # Data directory
├── models/           # Model checkpoints
└── logs/             # Training logs
```

## Development Workflow

1. **Feature Development**
   - Create a feature branch
   - Implement changes
   - Write tests
   - Update documentation
   - Create PR

2. **Bug Fixes**
   - Create a bug fix branch
   - Fix the issue
   - Add regression tests
   - Update documentation if needed
   - Create PR

3. **Documentation**
   - Create a docs branch
   - Update relevant documentation
   - Ensure examples are working
   - Create PR

## Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create a release branch
4. Run full test suite
5. Create a release PR
6. Tag the release
7. Build and publish to PyPI

## Getting Help

- Check the [documentation](https://github.com/SentiChain/aparecium/wiki)
- Open an issue for bugs or feature requests

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License. 