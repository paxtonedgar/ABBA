"""
Zero-Compromise Test Gauntlet Configuration

This conftest.py enforces the zero-mock mandate and provides real service fixtures
for end-to-end testing of the ABBA system.
"""

import asyncio
import os
import time
import tracemalloc
from collections.abc import AsyncGenerator, Generator

import pytest
import pytest_asyncio
import redis.asyncio as redis
import structlog
from asyncpg import create_pool
from docker import DockerClient
from docker.errors import DockerException
from httpx import AsyncClient
from testcontainers.core.container import DockerContainer
from testcontainers.core.waiting_utils import wait_for_logs
from testcontainers.postgres import PostgresContainer

# Configure logging for tests
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class ZeroMockEnforcer:
    """Enforces zero-mock policy during test execution."""

    def __init__(self):
        self.mock_usage = []
        self.test_start_time = None
        self.test_memory_start = None

    def start_test(self, test_name: str):
        """Start monitoring a test."""
        self.test_start_time = time.time()
        tracemalloc.start()
        self.test_memory_start = tracemalloc.get_traced_memory()
        logger.info("Starting test", test_name=test_name)

    def end_test(self, test_name: str):
        """End monitoring a test and check for violations."""
        if self.test_start_time:
            duration = time.time() - self.test_start_time
            if duration > 10:  # Unit test timeout
                pytest.fail(f"Test {test_name} exceeded 10s timeout: {duration:.2f}s")

        if self.test_memory_start:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            if peak > 100 * 1024 * 1024:  # 100MB limit
                pytest.fail(f"Test {test_name} exceeded memory limit: {peak / 1024 / 1024:.2f}MB")

        if self.mock_usage:
            pytest.fail(f"Test {test_name} used mocks: {self.mock_usage}")


@pytest.fixture(scope="session")
def zero_mock_enforcer() -> ZeroMockEnforcer:
    """Global enforcer for zero-mock policy."""
    return ZeroMockEnforcer()


@pytest.fixture(autouse=True)
def enforce_zero_mock(request, zero_mock_enforcer: ZeroMockEnforcer):
    """Automatically enforce zero-mock policy for all tests."""
    test_name = request.node.name
    zero_mock_enforcer.start_test(test_name)

    yield

    zero_mock_enforcer.end_test(test_name)


@pytest.fixture(scope="session")
def docker_client() -> DockerClient:
    """Real Docker client for container management."""
    try:
        client = DockerClient.from_env()
        client.ping()
        return client
    except DockerException as e:
        pytest.skip(f"Docker not available: {e}")


@pytest.fixture(scope="session")
def postgres_container(docker_client) -> Generator[PostgresContainer, None, None]:
    """Real PostgreSQL container for testing."""
    container = PostgresContainer(
        image="postgres:15-alpine",
        user="abba_test",
        password="abba_test_password",
        dbname="abba_test",
        port=5432
    )

    with container:
        wait_for_logs(container, "database system is ready to accept connections")
        yield container


@pytest.fixture(scope="session")
def redis_container(docker_client) -> Generator[DockerContainer, None, None]:
    """Real Redis container for testing."""
    container = DockerContainer("redis:7-alpine")
    container.with_exposed_ports(6379)

    with container:
        wait_for_logs(container, "Ready to accept connections")
        yield container


@pytest.fixture(scope="session")
def mock_api_container(docker_client) -> Generator[DockerContainer, None, None]:
    """Mock API server container for external service emulation."""
    container = DockerContainer("mockserver/mockserver:5.15")
    container.with_exposed_ports(1080)
    container.with_volume_mount(
        os.path.join(os.path.dirname(__file__), "mock-api-config"),
        "/config"
    )
    container.with_env("MOCKSERVER_PROPERTY_FILE", "/config/mockserver.properties")

    with container:
        wait_for_logs(container, "MockServer started on port")
        yield container


@pytest_asyncio.fixture(scope="session")
async def postgres_pool(postgres_container) -> AsyncGenerator:
    """Real PostgreSQL connection pool."""
    pool = await create_pool(
        host=postgres_container.get_container_host_ip(),
        port=postgres_container.get_exposed_port(5432),
        user="abba_test",
        password="abba_test_password",
        database="abba_test",
        min_size=1,
        max_size=10
    )

    # Initialize schema
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS bets (
                id SERIAL PRIMARY KEY,
                event_id VARCHAR(255) NOT NULL,
                sport VARCHAR(50) NOT NULL,
                odds DECIMAL(10,2) NOT NULL,
                stake DECIMAL(10,2) NOT NULL,
                expected_value DECIMAL(10,4) NOT NULL,
                confidence DECIMAL(5,4) NOT NULL,
                status VARCHAR(50) DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS analytics_results (
                id SERIAL PRIMARY KEY,
                model_name VARCHAR(255) NOT NULL,
                sport VARCHAR(50) NOT NULL,
                accuracy DECIMAL(5,4),
                precision DECIMAL(5,4),
                recall DECIMAL(5,4),
                f1_score DECIMAL(5,4),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    yield pool

    await pool.close()


@pytest_asyncio.fixture(scope="session")
async def redis_client(redis_container) -> AsyncGenerator:
    """Real Redis client."""
    client = redis.Redis(
        host=redis_container.get_container_host_ip(),
        port=redis_container.get_exposed_port(6379),
        decode_responses=True
    )

    # Test connection
    await client.ping()

    yield client

    await client.close()


@pytest_asyncio.fixture
async def http_client(mock_api_container) -> AsyncGenerator[AsyncClient, None]:
    """HTTP client for testing external APIs."""
    base_url = f"http://{mock_api_container.get_container_host_ip()}:{mock_api_container.get_exposed_port(1080)}"

    async with AsyncClient(base_url=base_url, timeout=30.0) as client:
        yield client


@pytest.fixture
def test_config(postgres_container, redis_container, mock_api_container):
    """Real test configuration with live services."""
    return {
        "database": {
            "url": f"postgresql://abba_test:abba_test_password@{postgres_container.get_container_host_ip()}:{postgres_container.get_exposed_port(5432)}/abba_test"
        },
        "redis": {
            "host": redis_container.get_container_host_ip(),
            "port": redis_container.get_exposed_port(6379)
        },
        "apis": {
            "mlb": {
                "base_url": f"http://{mock_api_container.get_container_host_ip()}:{mock_api_container.get_exposed_port(1080)}/api/v1"
            },
            "nhl": {
                "base_url": f"http://{mock_api_container.get_container_host_ip()}:{mock_api_container.get_exposed_port(1080)}/api/v1"
            },
            "openai": {
                "model": "gpt-3.5-turbo",
                "api_key": os.getenv("OPENAI_API_KEY", "test-key")
            }
        },
        "sports": [
            {"name": "baseball_mlb", "enabled": True},
            {"name": "hockey_nhl", "enabled": True}
        ],
        "platforms": {
            "fanduel": {"enabled": True},
            "draftkings": {"enabled": True}
        }
    }


@pytest.fixture(autouse=True)
def reset_database(postgres_pool):
    """Reset database state before each test."""
    async def _reset():
        async with postgres_pool.acquire() as conn:
            await conn.execute("TRUNCATE TABLE bets, analytics_results RESTART IDENTITY CASCADE")

    asyncio.run(_reset())


@pytest.fixture(autouse=True)
def reset_redis(redis_client):
    """Reset Redis state before each test."""
    async def _reset():
        await redis_client.flushdb()

    asyncio.run(_reset())


# Performance monitoring fixtures
@pytest.fixture(autouse=True)
def monitor_resources(request):
    """Monitor resource usage during tests."""

    import psutil

    process = psutil.Process()
    start_cpu = process.cpu_percent()
    start_memory = process.memory_info().rss
    start_threads = process.num_threads()
    start_fds = process.num_fds() if hasattr(process, 'num_fds') else 0

    yield

    end_cpu = process.cpu_percent()
    end_memory = process.memory_info().rss
    end_threads = process.num_threads()
    end_fds = process.num_fds() if hasattr(process, 'num_fds') else 0

    # Log resource usage
    logger.info(
        "Test resource usage",
        test_name=request.node.name,
        cpu_delta=end_cpu - start_cpu,
        memory_delta_mb=(end_memory - start_memory) / 1024 / 1024,
        thread_delta=end_threads - start_threads,
        fd_delta=end_fds - start_fds
    )

    # Fail if resource leaks detected
    if end_threads > start_threads + 5:  # More than 5 new threads
        pytest.fail(f"Thread leak detected: {end_threads - start_threads} new threads")

    if end_fds > start_fds + 10:  # More than 10 new file descriptors
        pytest.fail(f"File descriptor leak detected: {end_fds - start_fds} new FDs")


# Pytest configuration
def pytest_configure(config):
    """Configure pytest for zero-compromise testing."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to enforce zero-mock policy."""
    for item in items:
        # Add timeout to all tests
        item.add_marker(pytest.mark.timeout(30))

        # Mark integration tests as slow
        if "integration" in item.nodeid or "e2e" in item.nodeid:
            item.add_marker(pytest.mark.slow)


def pytest_runtest_setup(item):
    """Setup before each test to enforce zero-mock policy."""
    # Check for mock imports in test file
    test_file = item.fspath
    if test_file.exists():
        content = test_file.read_text()
        if any(mock_indicator in content for mock_indicator in [
            "unittest.mock", "pytest.mock", "@patch", "@mock", "Mock(", "MagicMock("
        ]):
            pytest.fail(f"Test file {test_file} contains mock usage - ZERO MOCK POLICY VIOLATION")


def pytest_runtest_teardown(item, nextitem):
    """Teardown after each test to check for resource leaks."""
    import gc
    gc.collect()  # Force garbage collection
