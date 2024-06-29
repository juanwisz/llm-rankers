import pytest
import pandas as pd
import json
import re
from collections import defaultdict
import random
from tqdm import tqdm

# Load the real inference cache
inference_cache = pd.read_pickle('inference_cache.pkl')

# Load the JSON file
with open('first_stage_rankings.json', 'r', encoding='utf-8') as f:
    json_data = json.load(f)

def extract_passages_from_key(key):
    """Extract passages from the key tuple using regex."""
    if not isinstance(key, tuple) or len(key) != 2:
        return None
    
    _, query_string = key
    pattern = r'Passage A: "(.*?)"\n\nPassage B: "(.*?)"'
    match = re.search(pattern, query_string, re.DOTALL)
    return match.groups() if match else None

def sample_dict(d, n=10000):
    """Sample a subset of a dictionary for faster processing."""
    keys = random.sample(list(d.keys()), min(n, len(d)))
    return {k: d[k] for k in keys}

@pytest.fixture(scope="module")
def passage_data():
    """Load and prepare all passage data once."""
    pkl_passages = set()
    json_passages = set()
    sampled_inference_cache = sample_dict(inference_cache)
    
    print("Extracting passages from PKL...")
    for key in tqdm(sampled_inference_cache.keys()):
        extracted = extract_passages_from_key(key)
        if extracted:
            pkl_passages.update(extracted)
    
    print("Extracting passages from JSON...")
    for item in tqdm(json_data):
        for doc in item[2]:  # item[2] contains the list of documents
            json_passages.add(doc['text'])
    
    return {
        'pkl': pkl_passages,
        'json': json_passages,
        'sampled_cache': sampled_inference_cache
    }

def test_passage_extraction_success_rate(passage_data):
    """Test the success rate of extracting passages from PKL keys."""
    total_keys = len(passage_data['sampled_cache'])
    successful_extractions = sum(1 for key in passage_data['sampled_cache'].keys() 
                                 if extract_passages_from_key(key) is not None)
    success_rate = successful_extractions / total_keys
    print(f"Passage extraction success rate: {success_rate:.2%}")
    assert success_rate > 0.95, f"Passage extraction success rate too low: {success_rate:.2%}"

def test_passages_in_json_not_in_pkl(passage_data):
    """Identify passages that are in the JSON but not in the PKL."""
    missing_passages = passage_data['json'] - passage_data['pkl']
    if missing_passages:
        print(f"Found {len(missing_passages)} passages in JSON but not in sampled PKL.")
        print("Sample of missing passages:")
        for passage in list(missing_passages)[:5]:  # Print first 5 for brevity
            print(f"- {passage[:100]}...")  # Print first 100 characters
    assert not missing_passages, f"{len(missing_passages)} passages found in JSON but not in sampled PKL"

def test_passages_in_pkl_not_in_json(passage_data):
    """Identify passages that are in the PKL but not in the JSON."""
    extra_passages = passage_data['pkl'] - passage_data['json']
    if extra_passages:
        print(f"Found {len(extra_passages)} passages in sampled PKL but not in JSON.")
        print("Sample of extra passages:")
        for passage in list(extra_passages)[:5]:  # Print first 5 for brevity
            print(f"- {passage[:100]}...")  # Print first 100 characters
    assert not extra_passages, f"{len(extra_passages)} passages found in sampled PKL but not in JSON"

def test_passage_length_distribution(passage_data):
    """Analyze the length distribution of passages in the PKL."""
    lengths = [len(passage) for passage in passage_data['pkl']]
    print("Passage length statistics (based on sampled PKL):")
    print(f"- Min: {min(lengths)}")
    print(f"- Max: {max(lengths)}")
    print(f"- Average: {sum(lengths) / len(lengths):.2f}")
    
    # Check for suspiciously short passages
    short_passages = [p for p in passage_data['pkl'] if len(p) < 50]  # Adjust threshold as needed
    if short_passages:
        print(f"Found {len(short_passages)} suspiciously short passages in sampled PKL:")
        for passage in short_passages[:5]:  # Print first 5 for brevity
            print(f"- {passage}")

def test_passage_content_patterns(passage_data):
    """Look for patterns or anomalies in passage content."""
    pattern_counts = defaultdict(int)
    for passage in passage_data['pkl']:
        if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', passage):
            pattern_counts['email'] += 1
        if re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', passage):
            pattern_counts['ip_address'] += 1
        if re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', passage):
            pattern_counts['url'] += 1
        if re.search(r'\b\d{10,}\b', passage):
            pattern_counts['long_number'] += 1
        if len(passage.split()) < 5:
            pattern_counts['very_short_passage'] += 1
    
    print("Content pattern statistics (based on sampled PKL):")
    for pattern, count in pattern_counts.items():
        print(f"- {pattern}: {count}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])