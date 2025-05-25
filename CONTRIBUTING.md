# Contributing to UAP Video Analysis Pipeline

Thank you for your interest in contributing to the UAP Video Analysis Pipeline! This project welcomes contributions from researchers, developers, and anyone interested in advancing the scientific analysis of aerial phenomena.

## 🤝 How to Contribute

### Reporting Issues
- Use the GitHub issue tracker to report bugs or request features
- Provide detailed information including:
  - Operating system and Python version
  - Hardware configuration (GPU, etc.)
  - Video format and characteristics
  - Steps to reproduce the issue
  - Expected vs actual behavior

### Feature Requests
- Suggest new analysis techniques or improvements
- Describe the scientific value or use case
- Consider providing research references if applicable

### Code Contributions
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes following the coding standards
4. Add tests for new functionality
5. Update documentation as needed
6. Submit a pull request

## 🧪 Development Setup

### Prerequisites
- Python 3.8+
- Git
- For GPU acceleration: CUDA toolkit (Linux) or Xcode (macOS)

### Setup Development Environment
```bash
# Clone your fork
git clone https://github.com/your-username/uap-video-analysis.git
cd uap-video-analysis

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install in development mode
pip install -e .

# Run tests
pytest tests/
```

## 📝 Coding Standards

### Python Style
- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Maximum line length: 88 characters (Black formatter)
- Use descriptive variable and function names

### Code Formatting
```bash
# Format code with Black
black src/ tests/

# Check with flake8
flake8 src/ tests/

# Sort imports
isort src/ tests/
```

### Documentation
- Use docstrings for all classes and functions
- Include parameter types and return values
- Provide usage examples for complex functions
- Update README.md for user-facing changes

## 🏗️ Architecture Guidelines

### Adding New Analyzers
When creating new analysis components:

1. **Create analyzer class** in `src/analyzers/`
2. **Implement standard interface**:
   ```python
   class NewAnalyzer:
       def __init__(self, config):
           """Initialize with configuration."""
           pass
       
       def analyze(self, frames, metadata):
           """Analyze frames and return results."""
           return {}
   ```
3. **Add configuration options** to `analysis_config.yaml`
4. **Update main pipeline** to include your analyzer
5. **Add tests** in `tests/test_analyzers.py`

### Hardware Acceleration
- Use the `AccelerationManager` for GPU operations
- Provide CPU fallbacks for all accelerated functions
- Test on both GPU and CPU configurations

### Performance Considerations
- Use progress bars for long operations
- Implement batch processing where possible
- Profile performance-critical sections
- Consider memory usage for large videos

## 🧪 Testing

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_analyzers.py

# Run with coverage
pytest --cov=src tests/
```

### Test Requirements
- All new features must have tests
- Maintain >80% code coverage
- Include both unit and integration tests
- Test edge cases and error conditions

### Test Data
- Use small, synthetic test videos when possible
- Avoid including large files in the repository
- Document any external test data requirements

## 📚 Research Contributions

### Scientific Accuracy
- Validate analysis methods against known phenomena
- Provide references for algorithms and techniques
- Consider statistical significance in pattern detection
- Document assumptions and limitations

### Documentation
- Explain the scientific basis for new analysis methods
- Include mathematical formulations where appropriate
- Provide examples of practical applications
- Consider creating tutorial notebooks

## 🔒 Security and Privacy

### Data Handling
- Never commit sensitive video content
- Respect privacy laws and regulations
- Provide clear guidance on data usage
- Consider anonymization techniques

### Code Security
- Validate all user inputs
- Use secure file handling practices
- Avoid executing arbitrary code
- Review dependencies for vulnerabilities

## 📋 Pull Request Process

1. **Before submitting**:
   - Ensure all tests pass
   - Update documentation
   - Add changelog entry
   - Verify platform compatibility

2. **Pull request description**:
   - Clearly describe the changes
   - Reference related issues
   - Include screenshots/examples if applicable
   - Note any breaking changes

3. **Review process**:
   - Maintainers will review your code
   - Address feedback promptly
   - Be open to suggestions and improvements
   - Squash commits if requested

## 🌟 Recognition

Contributors will be acknowledged in:
- CONTRIBUTORS.md file
- Release notes for significant contributions
- Research publications (where appropriate)
- Project documentation

## 📞 Getting Help

- Join discussions in GitHub Issues
- Ask questions in pull requests
- Contact maintainers for major contributions
- Check existing documentation and examples

## 📄 License

By contributing, you agree that your contributions will be licensed under the same MIT License that covers the project. See LICENSE file for details.

---

**Thank you for helping advance UAP research through better analysis tools!** 🛸