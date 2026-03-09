# ABBA - Advanced Baseball Betting Analytics

A comprehensive sports betting analytics platform focused on MLB and NHL, featuring advanced machine learning, biometric integration, and real-time data processing.

## Features

- **Ensemble ML Models**: Random forest and gradient boosting ensembles with biometric feature integration
- **Real-time Odds Analysis**: Live sports data processing and arbitrage detection
- **Personalization Engine**: User-specific model training and pattern analysis
- **Graph Analysis**: Team performance modeling using network theory
- **Browser Automation**: Automated data collection and betting execution via BrowserBase
- **Fund Management**: Bankroll management with configurable risk tolerance

## Tech Stack

- **Backend**: Python (analytics, ML, agents), Go (services)
- **ML**: Scikit-learn ensembles, biometric feature engineering, personalization models
- **Data**: SQLite, Redis, NumPy/Pandas
- **Automation**: BrowserBase, BrightData
- **Quality**: Black, Ruff, MyPy, pytest

## Project Structure

```
Projects/ABBA/
├── src/abba/                 # Main package
│   ├── core/                 # Config, logging
│   ├── analytics/            # ML models, biometrics, ensemble, graph analysis
│   ├── trading/              # Trading algorithms
│   ├── agents/               # AI agents
│   ├── data/                 # Data pipeline
│   └── api/                  # API endpoints
├── tests/                    # Unit + integration tests
├── docs/                     # Strategy docs, system design
└── examples/                 # Usage examples
```

## Quick Start

```bash
git clone https://github.com/paxtonedgar/ABBA.git
cd ABBA/Projects/ABBA

pip install -e .
pip install -e ".[dev]"
```

Create a `.env` file:

```env
OPENAI_API_KEY=your_key
BROWSERBASE_API_KEY=your_key
DATABASE_URL=sqlite:///abba.db
REDIS_URL=redis://localhost:6379
SUPPORTED_SPORTS=MLB,NHL
```

## Usage

```python
from abba import AdvancedAnalyticsManager
from abba.core import Config

config = Config()
analytics = AdvancedAnalyticsManager(config, db_manager)

# Ensemble prediction
ensemble = await analytics.create_ensemble_model(['random_forest', 'gradient_boosting'])
prediction = await analytics.ensemble_predictions(ensemble, features)

# Biometric integration
features = await analytics.integrate_biometrics(biometric_data)
```

## Testing

```bash
pytest
pytest --cov=src/abba --cov-report=html
pytest -m unit
pytest -m integration
```

## Documentation

See the [docs/](Projects/ABBA/docs/) directory for detailed documentation on MLB/NHL strategies, data pipeline architecture, fund management, and system design.

## License

MIT
