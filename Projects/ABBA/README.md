# ABBA - Advanced Baseball Betting Analytics

A comprehensive sports betting analytics platform focused on MLB and NHL, featuring advanced machine learning, biometric integration, and real-time data processing.

## 🚀 Features

- **Advanced Analytics**: Ensemble machine learning models with biometric integration
- **Real-time Data**: Live sports data processing and odds analysis
- **Personalization**: User-specific model training and pattern analysis
- **Graph Analysis**: Team performance analysis using network theory
- **Automation**: Browser automation for data collection and betting execution
- **API Integration**: RESTful API for external integrations

## 📚 Documentation

### Core Documentation
- **[Project Specification](docs/PROJECT_SPECIFICATION.md)** - Comprehensive system overview and requirements
- **[System Analysis](docs/system-analysis.md)** - Architecture and design decisions
- **[Implementation Plans](docs/implementation-plans.md)** - Development roadmap and phases

### Sports Strategies
- **[MLB Strategy](docs/mlb-strategy.md)** - Baseball betting strategies and analysis
- **[NHL Strategy](docs/nhl-strategy.md)** - Hockey betting strategies and analysis

### Technical Integration
- **[Data Pipeline](docs/data-pipeline.md)** - Data processing and analytics pipeline
- **[BrowserBase Integration](docs/browserbase-integration.md)** - Web automation setup
- **[BrightData Integration](docs/brightdata-integration.md)** - Data collection services
- **[Database Setup](docs/database-setup.md)** - Database configuration and management

### Operations & Management
- **[Fund Management](docs/fund-management.md)** - Bankroll management and risk control
- **[Anti-Detection Security](docs/anti-detection-security.md)** - Stealth and security measures
- **[Demo & Live Testing](docs/demo-live-testing.md)** - Testing procedures and validation
- **[Professional Analytics](docs/professional-analytics.md)** - Advanced analytics features

### Development & Testing
- **[Validation Testing](docs/validation-testing.md)** - Testing framework and procedures
- **[Debugging Guide](docs/debugging.md)** - Troubleshooting and debugging
- **[Balance Monitoring](docs/BALANCE_MONITORING_SUMMARY.md)** - Account balance tracking

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- pip or poetry

### Quick Start

```bash
# Clone the repository
git clone https://github.com/abba-team/abba.git
cd abba

# Install dependencies
pip install -e .

# Install development dependencies
pip install -e ".[dev]"

# Setup pre-commit hooks
pre-commit install
```

### Environment Setup

Create a `.env` file in the project root:

```env
# API Keys
OPENAI_API_KEY=your_openai_key_here
BROWSERBASE_API_KEY=your_browserbase_key_here

# Database
DATABASE_URL=sqlite:///abba.db
REDIS_URL=redis://localhost:6379

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/abba.log

# Sports Configuration
SUPPORTED_SPORTS=MLB,NHL

# Trading Configuration
MAX_BET_AMOUNT=100.0
RISK_TOLERANCE=0.1
```

## 🏗️ Project Structure

```
abba/
├── src/abba/                 # Main package
│   ├── core/                 # Core functionality
│   │   ├── config.py        # Configuration management
│   │   └── logging.py       # Logging setup
│   ├── analytics/           # Analytics modules
│   │   ├── manager.py       # Analytics manager
│   │   ├── biometrics.py    # Biometric processing
│   │   ├── personalization.py # User personalization
│   │   ├── ensemble.py      # Ensemble methods
│   │   ├── graph.py         # Graph analysis
│   │   └── models.py        # Data models
│   ├── trading/             # Trading algorithms
│   ├── agents/              # AI agents
│   ├── data/                # Data processing
│   ├── utils/               # Utilities
│   └── api/                 # API endpoints
├── tests/                   # Test suite
│   ├── unit/               # Unit tests
│   └── integration/        # Integration tests
├── examples/               # Example scripts
├── docs/                   # Documentation
├── pyproject.toml          # Project configuration
├── requirements-dev.txt    # Development dependencies
└── README.md              # This file
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/abba --cov-report=html

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m "not slow"
```

## 🔧 Development

### Code Quality

The project uses modern Python development tools:

- **Black**: Code formatting
- **Ruff**: Linting and import sorting
- **MyPy**: Type checking
- **Pre-commit**: Git hooks for code quality

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type check
mypy src/

# Run all quality checks
pre-commit run --all-files
```

### Adding New Features

1. Create feature branch: `git checkout -b feature/new-feature`
2. Implement feature with tests
3. Run quality checks: `pre-commit run --all-files`
4. Submit pull request

## 📊 Usage Examples

### Basic Analytics

```python
from abba import AdvancedAnalyticsManager
from abba.core import Config

# Initialize
config = Config()
analytics = AdvancedAnalyticsManager(config, db_manager)

# Process biometric data
biometric_data = {
    'heart_rate': [75, 78, 82, 79, 76],
    'fatigue_metrics': {'sleep_quality': 0.8, 'stress_level': 0.3},
    'movement': {'total_distance': 5000, 'avg_speed': 2.1}
}

features = await analytics.integrate_biometrics(biometric_data)
```

### Ensemble Predictions

```python
# Create ensemble model
ensemble = await analytics.create_ensemble_model([
    'random_forest', 'gradient_boosting'
])

# Make prediction
prediction = await analytics.ensemble_predictions(ensemble, features)
print(f"Prediction: {prediction.value:.4f} ± {prediction.error_margin:.4f}")
```

### Personalization

```python
# Analyze user patterns
user_history = [...]  # User's betting history
patterns = await analytics.personalize_models(user_history)

# Create personalized model
personalized_model = await analytics.personalization_engine.create_model(patterns)
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
4. Install in development mode: `pip install -e ".[dev]"`
5. Run tests: `pytest`

### Code Style

- Follow PEP 8 guidelines
- Use type hints for all functions
- Write docstrings for all public functions
- Keep functions under 50 lines when possible
- Use meaningful variable names

## 📈 Performance

### Benchmarks

- **Model Training**: < 30 seconds for standard ensemble
- **Prediction Latency**: < 100ms for real-time predictions
- **Data Processing**: 1000+ records/second
- **Memory Usage**: < 2GB for typical workloads

### Optimization

- Use async/await for I/O operations
- Implement caching for expensive computations
- Batch process data when possible
- Use vectorized operations with NumPy

## 🔒 Security

- API keys stored in environment variables
- Input validation on all endpoints
- Rate limiting for API calls
- Secure database connections
- Regular security audits

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [https://abba.readthedocs.io](https://abba.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/abba-team/abba/issues)
- **Discussions**: [GitHub Discussions](https://github.com/abba-team/abba/discussions)
- **Email**: support@abba.com

## 🙏 Acknowledgments

- Sports data providers
- Open source community
- Beta testers and contributors

--- 