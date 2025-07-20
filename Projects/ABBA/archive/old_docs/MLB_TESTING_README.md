# MLB Prediction System Testing Suite

A comprehensive testing framework for MLB prediction models using real API data. This system tests all games in the MLB season with robust statistical validation and performance benchmarks.

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** with pip
2. **API Keys** (optional but recommended for full testing):
   - The Odds API key (for real odds data)
   - Sports Data API key (for additional sports data)

### Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set up environment variables** (optional):
```bash
export ODDS_API_KEY="your_odds_api_key"
export SPORTS_DATA_API_KEY="your_sports_data_api_key"
```

### Running Tests

#### Quick Test (Recommended for first run)
```bash
python test_mlb_system.py --quick
```

#### Full Comprehensive Test
```bash
python test_mlb_system.py --full
```

#### Test Specific Season
```bash
python test_mlb_system.py --season 2024
```

#### Verbose Mode
```bash
python test_mlb_system.py --full --verbose
```

## 📊 What Gets Tested

### 1. **Data Quality Tests**
- **Completeness**: Ensures data coverage is at least 80%
- **Anomaly Detection**: Identifies outliers and data inconsistencies
- **Physics Validation**: Validates baseball-specific metrics (exit velocity, spin rate, etc.)
- **API Reliability**: Tests response times and success rates

### 2. **Prediction Accuracy Tests**
- **Consistency**: Ensures predictions are stable across multiple runs
- **Probability Constraints**: Validates that home/away probabilities sum to ~1.0
- **Confidence Bounds**: Checks that confidence values are within reasonable ranges
- **Model Performance**: Tests XGBoost and ensemble model accuracy

### 3. **API Performance Tests**
- **Response Times**: Ensures APIs respond within 30 seconds
- **Success Rates**: Validates API reliability (target: >80%)
- **Rate Limiting**: Tests behavior under rapid requests
- **Data Freshness**: Checks for recent and relevant data

### 4. **Integration Tests**
- **End-to-End Pipeline**: Tests complete workflow from data fetch to prediction
- **Feature Engineering**: Validates feature creation and processing
- **Model Training**: Tests model training performance and accuracy
- **Data Flow**: Ensures seamless data movement between components

### 5. **Performance Benchmarks**
- **Data Fetching Speed**: Records per second metrics
- **Feature Engineering**: Processing time and feature generation rate
- **Prediction Speed**: Time per prediction
- **Memory Usage**: Resource consumption tracking

## 📁 Test Files Structure

```
├── mlb_season_testing_system.py    # Main testing system
├── mlb_prediction_test_suite.py    # Pytest test suite
├── run_mlb_tests.py               # Advanced test runner
├── test_mlb_system.py             # Simple test runner
└── results/
    └── mlb_season_tests/          # Test results storage
```

## 🔍 Understanding Test Results

### Test Status Levels

- **EXCELLENT**: >80% test coverage, >70% data quality, >70% performance
- **GOOD**: >60% test coverage, >50% data quality, >50% performance  
- **FAIR**: >40% test coverage
- **POOR**: <40% test coverage or critical failures

### Key Metrics

1. **Test Coverage**: Percentage of tests that passed
2. **Data Quality Score**: Overall data reliability (0-1 scale)
3. **Performance Score**: System efficiency (0-1 scale)
4. **API Success Rate**: Percentage of successful API calls
5. **Prediction Confidence**: Average confidence in predictions

### Sample Output

```
============================================================
📊 MLB PREDICTION SYSTEM TEST SUMMARY
============================================================
Overall Status: GOOD
Test Coverage: 75.0%
Data Quality Score: 0.823
Performance Score: 0.733

Test Results:
  Total Tests: 24
  Passed: 18
  Failed: 2
  Skipped: 4

Total Duration: 45.23 seconds

Recommendations:
  • Increase test coverage to at least 80%
  • System is performing well - continue monitoring
============================================================
```

## 🛠️ Advanced Usage

### Custom Configuration

Create a custom `config.yaml`:
```yaml
apis:
  odds_api:
    key: "${ODDS_API_KEY}"
    rate_limit: 500
  sports_data_io:
    key: "${SPORTS_DATA_API_KEY}"

sports:
  - name: "baseball_mlb"
    enabled: true
    markets: ["moneyline", "spread", "totals"]
```

### Running Specific Test Categories

```bash
# Run only data quality tests
python -m pytest mlb_prediction_test_suite.py::TestMLBDataQuality -v

# Run only API performance tests  
python -m pytest mlb_prediction_test_suite.py::TestMLBAPIPerformance -v

# Run only prediction accuracy tests
python -m pytest mlb_prediction_test_suite.py::TestMLBPredictionAccuracy -v
```

### Continuous Integration

Add to your CI pipeline:
```yaml
# GitHub Actions example
- name: Run MLB Tests
  run: |
    python test_mlb_system.py --quick
    python test_mlb_system.py --full --verbose
```

## 📈 Interpreting Results

### Data Quality Issues

**Low Coverage (<80%)**: 
- Check API connectivity
- Verify data sources are active
- Review data fetching logic

**High Anomaly Rate (>10%)**:
- Investigate data source issues
- Check for sensor malfunctions
- Review data cleaning procedures

### Performance Issues

**Slow API Response (>30s)**:
- Check network connectivity
- Verify API rate limits
- Consider caching strategies

**Low Prediction Speed (>5s)**:
- Optimize feature engineering
- Review model complexity
- Consider model caching

### Accuracy Issues

**Low Test Coverage (<60%)**:
- Add more comprehensive tests
- Fix failing test cases
- Improve error handling

**Poor Model Performance (<40% accuracy)**:
- Review feature engineering
- Check training data quality
- Consider model retraining

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**:
   ```bash
   pip install -r requirements.txt
   ```

2. **API Key Issues**:
   ```bash
   export ODDS_API_KEY="your_key_here"
   ```

3. **Permission Errors**:
   ```bash
   chmod +x test_mlb_system.py
   ```

4. **Memory Issues**:
   - Reduce test data size
   - Use `--quick` mode
   - Increase system memory

### Debug Mode

Enable verbose logging:
```bash
python test_mlb_system.py --full --verbose
```

Check logs in `logs/` directory for detailed error information.

## 📊 Test Results Analysis

### Performance Benchmarks

The system tracks:
- **Data Fetching**: Records per second
- **Feature Engineering**: Features per second  
- **Prediction Speed**: Predictions per second
- **Memory Usage**: Peak memory consumption

### Statistical Validation

Tests include:
- **Z-score Analysis**: Outlier detection
- **Isolation Forest**: Anomaly detection
- **Physics Validation**: Baseball-specific constraints
- **Probability Validation**: Mathematical consistency

### Quality Metrics

- **Completeness**: Data coverage percentage
- **Consistency**: Cross-validation results
- **Reliability**: API success rates
- **Accuracy**: Model performance metrics

## 🎯 Best Practices

1. **Run Quick Tests First**: Always start with `--quick` to verify basic functionality
2. **Monitor API Limits**: Be aware of rate limits when running full tests
3. **Check Results Directory**: Review detailed results in `results/mlb_season_tests/`
4. **Use Version Control**: Track test results over time to identify trends
5. **Set Up Alerts**: Monitor for test failures in production environments

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the test logs in `logs/` directory
3. Examine detailed results in `results/mlb_season_tests/`
4. Run with `--verbose` flag for detailed output

## 🔄 Continuous Testing

For ongoing monitoring, consider:
- **Scheduled Tests**: Run tests daily/weekly
- **Automated Alerts**: Set up notifications for test failures
- **Trend Analysis**: Track performance over time
- **Regression Testing**: Compare against baseline results

---

**Note**: This testing system is designed to work with real MLB data APIs. Some tests may be skipped if APIs are unavailable or rate-limited. The system gracefully handles these scenarios and provides fallback mock data for testing. 