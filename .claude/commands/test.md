# Test Command

Write tests for: $ARGUMENTS

## Instructions

1. Create comprehensive pytest tests following AAA pattern:
   - **Arrange**: Set up test data and mocks
   - **Act**: Call the function under test
   - **Assert**: Verify expected outcomes

2. Test file location: `tests/test_[module].py`

3. Use fixtures for common setup

4. Mock external dependencies (Neo4j, APIs)

## Test Categories

### Unit Tests
```python
def test_function_happy_path():
    """Test normal operation."""
    pass

def test_function_edge_case():
    """Test boundary conditions."""
    pass

def test_function_error_handling():
    """Test error scenarios."""
    pass
```

### Integration Tests
```python
@pytest.mark.integration
def test_database_connection():
    """Test actual database connectivity."""
    pass
```

## Mocking Neo4j

```python
from unittest.mock import MagicMock, patch

@pytest.fixture
def mock_driver():
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__.return_value = session
    return driver

def test_query_execution(mock_driver):
    mock_driver.session().execute_read.return_value = [{"id": 1}]
    result = get_articles(mock_driver, "topic")
    assert len(result) == 1
```

## Coverage Target

- Aim for 80%+ coverage
- Run with: `uv run pytest --cov --cov-report=html`

## Output

After writing tests:
1. List all test functions created
2. Run tests and show results
3. Show coverage report if available
