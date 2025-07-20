# Zero-Compromise Test Gauntlet

A production-grade testing framework that enforces zero mocking, real service dependencies, and end-to-end validation for the ABBA sports betting analytics platform.

## 🎯 Mission

Transform the entire test layer into an unforgiving, real-world validation gauntlet where:
- **No mocks, stubs, fakes, or monkey-patches** are allowed
- Every test exercises **actual code paths, real I/O, real network, real databases**
- Tests pass only if the system behaves correctly under **production-like conditions**
- Any code that cannot be tested end-to-end must be **refactored until it can**

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- 8GB+ RAM (for running multiple services)

### Run the Complete Gauntlet

```bash
# From the project root
./run_test_gauntlet.sh
```

This script will:
1. ✅ Check prerequisites
2. 🐳 Start all required services (PostgreSQL, Redis, Mock API)
3. 🧪 Run the complete test suite with real services
4. 🧬 Execute mutation testing
5. 📊 Generate coverage and audit reports
6. 🧹 Clean up all resources

### Manual Setup

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Start services
cd tests
docker-compose up -d

# Wait for services to be ready
docker-compose exec postgres pg_isready -U abba_test -d abba_test
docker-compose exec redis redis-cli ping
curl -f http://localhost:1080/status

# Run tests
pytest -v -n auto --cov=src --cov-branch --timeout=30 \
    test_analytics_zero_mock.py test_database_zero_mock.py test_api_zero_mock.py

# Run mutation testing
mutmut run --paths-to-mutate src/abba/analytics/ src/abba/core/ src/abba/agents_modules/
```

## 🏗️ Architecture

### Test Environment

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │      Redis      │    │   Mock API      │
│   (Real DB)     │    │   (Real Cache)  │    │ (External APIs) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Test Suite    │
                    │ (Zero Mocks)    │
                    └─────────────────┘
```

### Service Configuration

| Service | Port | Purpose | Real/Emulated |
|---------|------|---------|---------------|
| PostgreSQL | 5432 | Database | Real |
| Redis | 6379 | Cache | Real |
| MockServer | 1080 | External APIs | Emulated |
| LocalStack | 4566 | AWS Services | Emulated |

## 📋 Test Categories

### 1. Analytics Tests (`test_analytics_zero_mock.py`)
- **Real XGBoost model training** with actual data
- **Live database storage** of model results
- **Redis cache operations** for performance
- **Feature engineering** with real MLB/NHL data
- **Model persistence** and retrieval

### 2. Database Tests (`test_database_zero_mock.py`)
- **Real PostgreSQL transactions** with ACID compliance
- **Data integrity constraints** (foreign keys, unique constraints)
- **Concurrent operations** testing
- **Connection pool management**
- **Schema migrations** and data validation

### 3. API Tests (`test_api_zero_mock.py`)
- **Real HTTP client** communication
- **External API integration** with MockServer
- **Error handling** and rate limiting
- **Authentication** and headers
- **Performance testing** under load

## 🔒 Quality Gates

### Zero Mocking Enforcement
- ❌ `unittest.mock` - BANNED
- ❌ `pytest-mock` - BANNED
- ❌ `@patch` decorators - BANNED
- ❌ `Mock()` objects - BANNED
- ✅ Real service containers - REQUIRED

### Coverage Requirements
- **Line Coverage**: ≥95%
- **Branch Coverage**: ≥90%
- **Mutation Score**: ≥80%
- **Test Runtime**: ≤15 minutes

### Performance Limits
- **Individual Test Timeout**: 30 seconds
- **Memory Usage**: ≤100MB per test
- **Resource Leaks**: 0 tolerance
- **Flaky Tests**: 0 tolerance

## 🛠️ Configuration

### Environment Variables

```bash
# Required for tests
export PYTHONPATH="${PYTHONPATH:-.}:src"
export OPENAI_API_KEY="${OPENAI_API_KEY:-test-key}"

# AWS services (LocalStack)
export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export AWS_DEFAULT_REGION=us-east-1

# Database
export POSTGRES_DB=abba_test
export POSTGRES_USER=abba_test
export POSTGRES_PASSWORD=abba_test_password
```

### Docker Compose Services

```yaml
# tests/docker-compose.yml
services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: abba_test
      POSTGRES_USER: abba_test
      POSTGRES_PASSWORD: abba_test_password
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  mock-api:
    image: mockserver/mockserver:5.15
    ports:
      - "1080:1080"
```

## 📊 Monitoring & Observability

### Resource Monitoring
- **Memory Usage**: Tracked per test with tracemalloc
- **Thread Count**: Monitored for leaks
- **File Descriptors**: Counted and validated
- **CPU Usage**: Measured and logged

### Performance Metrics
- **Response Times**: API endpoint performance
- **Database Queries**: Query count and duration
- **Cache Hit Rates**: Redis performance metrics
- **Test Execution Time**: Individual and suite timing

### Logging
- **Structured Logs**: JSON format with structlog
- **Test Context**: Each test logs its execution details
- **Error Tracking**: Comprehensive error capture
- **Resource Usage**: Memory, CPU, and I/O metrics

## 🔄 CI/CD Integration

### GitHub Actions Workflow

The test gauntlet is integrated into CI/CD with:

```yaml
# .github/workflows/test-gauntlet.yml
name: Zero-Compromise Test Gauntlet
on: [push, pull_request]

jobs:
  test-gauntlet:
    services:
      postgres: # Real PostgreSQL service
      redis:    # Real Redis service
      mock-api: # MockServer for APIs
    steps:
      - name: Run Test Suite
        run: |
          pytest -v -n auto --cov=src --cov-branch \
            --cov-fail-under=95 --timeout=30
      
      - name: Mutation Testing
        run: mutmut run --paths-to-mutate src/
      
      - name: Generate Audit Report
        run: python generate_audit_report.py
```

### Quality Gates in CI
- **Coverage Threshold**: 95% line, 90% branch
- **Mutation Score**: ≥80%
- **Test Timeout**: 30 seconds per test
- **Resource Limits**: Enforced per test
- **Mock Detection**: Automatic failure on mock usage

## 🧪 Writing Zero-Mock Tests

### Example: Real Database Test

```python
@pytest.mark.integration
async def test_real_database_operations(self, postgres_pool):
    """Test real database operations with zero mocks."""
    
    # Real database connection
    async with postgres_pool.acquire() as conn:
        # Real INSERT operation
        await conn.execute("""
            INSERT INTO test_events (event_id, sport, home_team, away_team)
            VALUES ($1, $2, $3, $4)
        """, "test_001", "mlb", "Yankees", "Red Sox")
        
        # Real SELECT operation
        result = await conn.fetchrow("""
            SELECT * FROM test_events WHERE event_id = $1
        """, "test_001")
        
        # Real assertions
        assert result['sport'] == "mlb"
        assert result['home_team'] == "Yankees"
```

### Example: Real API Test

```python
@pytest.mark.integration
async def test_real_api_integration(self, http_client):
    """Test real API integration with zero mocks."""
    
    # Real HTTP request
    response = await http_client.get("/api/v1/statcast", params={
        "date": "2024-01-01"
    })
    
    # Real response validation
    assert response.status_code == 200
    data = response.json()
    assert len(data) > 0
    
    # Real data validation
    first_record = data[0]
    assert "player_name" in first_record
    assert isinstance(first_record["release_speed"], (int, float))
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Docker Services Not Starting
```bash
# Check Docker status
docker info

# Check service logs
docker-compose logs postgres
docker-compose logs redis
docker-compose logs mock-api

# Restart services
docker-compose down --volumes
docker-compose up -d
```

#### 2. Test Timeouts
```bash
# Increase timeout for debugging
pytest --timeout=60

# Check resource usage
docker stats

# Monitor test execution
pytest -v --durations=10
```

#### 3. Coverage Below Threshold
```bash
# Generate detailed coverage report
pytest --cov=src --cov-report=html --cov-report=term-missing

# Check uncovered lines
open htmlcov/index.html
```

#### 4. Mock Detection False Positives
```bash
# Check for mock usage
grep -r "unittest.mock\|pytest.mock\|@patch\|Mock(" tests/

# Verify zero-mock compliance
pytest --collect-only
```

### Performance Optimization

#### 1. Parallel Execution
```bash
# Use all CPU cores
pytest -n auto --dist=worksteal

# Limit to specific number
pytest -n 4
```

#### 2. Service Optimization
```bash
# Increase PostgreSQL connections
POSTGRES_MAX_CONNECTIONS=100 docker-compose up

# Optimize Redis memory
REDIS_MAXMEMORY=512mb docker-compose up
```

#### 3. Test Selection
```bash
# Run only fast tests
pytest -m "not slow"

# Run only integration tests
pytest -m integration

# Run specific test file
pytest test_analytics_zero_mock.py
```

## 📈 Metrics & Reporting

### Coverage Reports
- **HTML Report**: `htmlcov/index.html`
- **XML Report**: `coverage.xml` (for CI integration)
- **Terminal Report**: Real-time coverage display

### Mutation Testing
- **Results**: `mutation_results.json`
- **Score**: Percentage of mutations killed
- **Details**: Specific mutations and their status

### Performance Reports
- **Benchmarks**: pytest-benchmark results
- **Memory Profiling**: pytest-memray reports
- **Resource Usage**: Per-test resource tracking

### Audit Reports
- **Compliance**: Mandate compliance checklist
- **Metrics**: Before/after comparisons
- **Recommendations**: Improvement suggestions

## 🔮 Future Enhancements

### Planned Features
1. **Load Testing**: Real performance testing under load
2. **Chaos Engineering**: Service failure simulation
3. **Contract Testing**: API contract validation
4. **Visual Regression**: UI testing with real browsers
5. **Security Testing**: Vulnerability scanning integration

### Maintenance
1. **Regular Updates**: Keep dependencies current
2. **Coverage Monitoring**: Maintain high coverage standards
3. **Performance Tracking**: Monitor test suite performance
4. **Security Scanning**: Regular vulnerability assessments

## 📚 Additional Resources

- [Zero-Compromise Testing Principles](https://martinfowler.com/articles/microservice-testing/)
- [Real Service Testing](https://www.testcontainers.org/)
- [Mutation Testing Guide](https://mutmut.readthedocs.io/)
- [Pytest Best Practices](https://docs.pytest.org/en/stable/)

## 🤝 Contributing

When adding new tests to the gauntlet:

1. **No Mocks**: Use real services only
2. **Real Data**: Generate realistic test data
3. **End-to-End**: Test complete workflows
4. **Performance**: Monitor resource usage
5. **Documentation**: Update this README

## 📄 License

This test gauntlet is part of the ABBA project and follows the same licensing terms.

---

**Remember**: The goal is not just to test, but to validate that the system works correctly under production-like conditions with zero compromises on quality or realism. 