import pytest
from collections import defaultdict
import re
import random
import string

# Mock inference cache for testing
mock_inference_cache = {
    ('model_name', 'Given a query "test query", which of the following two passages is more relevant to the query?\n\nPassage A: "This is passage A"\n\nPassage B: "This is passage B"\n\nOutput Passage A or Passage B:'): "Passage A",
    ('model_name', 'Given a query "another query", which of the following two passages is more relevant to the query?\n\nPassage A: "Passage A content"\n\nPassage B: "Passage B content"\n\nOutput Passage A or Passage B:'): "Passage B"
}

def parse_key(key):
    """
    Parse the complex key structure.
    
    :param key: Tuple containing model name and query string
    :return: Dictionary with parsed components
    """
    model_name, query_string = key
    pattern = r'Given a query "(.*?)", which of the following two passages is more relevant to the query\?\n\nPassage A: "(.*?)"\n\nPassage B: "(.*?)"\n\nOutput Passage A or Passage B:'
    match = re.match(pattern, query_string)
    if match:
        return {
            'model_name': model_name,
            'query': match.group(1),
            'passage_a': match.group(2),
            'passage_b': match.group(3)
        }
    return None

def generate_mock_key(model_name, query, passage_a, passage_b):
    """
    Generate a mock key for testing.
    
    :param model_name: Name of the model
    :param query: Query string
    :param passage_a: Content of Passage A
    :param passage_b: Content of Passage B
    :return: Tuple representing the key
    """
    query_string = f'Given a query "{query}", which of the following two passages is more relevant to the query?\n\nPassage A: "{passage_a}"\n\nPassage B: "{passage_b}"\n\nOutput Passage A or Passage B:'
    return (model_name, query_string)

def generate_random_string(length):
    """Generate a random string of given length."""
    return ''.join(random.choices(string.ascii_letters + string.digits + string.punctuation + ' ', k=length))

@pytest.fixture
def sample_inference_cache():
    """Fixture to provide a sample inference cache for testing."""
    cache = {}
    models = ['model1', 'model2', 'model3']
    for _ in range(100):
        model = random.choice(models)
        query = generate_random_string(20)
        passage_a = generate_random_string(50)
        passage_b = generate_random_string(50)
        key = generate_mock_key(model, query, passage_a, passage_b)
        cache[key] = random.choice(['Passage A', 'Passage B'])
    return cache

def test_parse_key():
    """Test the parse_key function."""
    for key in mock_inference_cache.keys():
        parsed = parse_key(key)
        assert parsed is not None
        assert 'model_name' in parsed
        assert 'query' in parsed
        assert 'passage_a' in parsed
        assert 'passage_b' in parsed

def test_generate_mock_key():
    """Test the generate_mock_key function."""
    key = generate_mock_key('test_model', 'test query', 'passage A', 'passage B')
    assert isinstance(key, tuple)
    assert len(key) == 2
    assert key[0] == 'test_model'
    assert 'test query' in key[1]
    assert 'passage A' in key[1]
    assert 'passage B' in key[1]

def test_key_structure(sample_inference_cache):
    """Test the structure of keys in the inference cache."""
    for key in sample_inference_cache.keys():
        assert isinstance(key, tuple)
        assert len(key) == 2
        assert isinstance(key[0], str)
        assert isinstance(key[1], str)

def test_value_structure(sample_inference_cache):
    """Test the structure of values in the inference cache."""
    for value in sample_inference_cache.values():
        assert value in ['Passage A', 'Passage B']

def test_query_extraction(sample_inference_cache):
    """Test extraction of queries from the inference cache."""
    queries = set()
    for key in sample_inference_cache.keys():
        parsed = parse_key(key)
        if parsed:
            queries.add(parsed['query'])
    assert len(queries) > 0

def test_passage_extraction(sample_inference_cache):
    """Test extraction of passages from the inference cache."""
    passages = defaultdict(set)
    for key in sample_inference_cache.keys():
        parsed = parse_key(key)
        if parsed:
            passages['A'].add(parsed['passage_a'])
            passages['B'].add(parsed['passage_b'])
    assert len(passages['A']) > 0
    assert len(passages['B']) > 0

def test_model_distribution(sample_inference_cache):
    """Test the distribution of models in the inference cache."""
    models = defaultdict(int)
    for key in sample_inference_cache.keys():
        models[key[0]] += 1
    assert len(models) > 1

def test_regenerate_key():
    """Test regenerating a key from parsed components."""
    original_key = next(iter(mock_inference_cache.keys()))
    parsed = parse_key(original_key)
    regenerated_key = generate_mock_key(parsed['model_name'], parsed['query'], parsed['passage_a'], parsed['passage_b'])
    assert original_key == regenerated_key

def test_key_uniqueness(sample_inference_cache):
    """Test that all keys in the inference cache are unique."""
    keys = list(sample_inference_cache.keys())
    assert len(keys) == len(set(keys))

if __name__ == "__main__":
    pytest.main([__file__])