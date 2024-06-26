# %% [markdown]
# We download the cache, this allows us to try out lots of different sorting algorithms virtually for free.

# %%
from collections import defaultdict
import pandas as pd
import gdown
import requests
import json
import subprocess
import tempfile
from typing import Dict, List
from tqdm import tqdm
from abc import ABC, abstractmethod
import re
from pyserini.util import download_evaluation_script
import os
# URL del archivo en Google Drive
url = 'https://drive.google.com/uc?export=download&id=1BNvGwclCRhRxoq_motDzsmDQO6ZFgx7b'

# Descargar el archivo
output = 'inference_cache.pkl'
# Verificar si el archivo ya existe
if not os.path.exists(output):
    # Descargar el archivo si no existe
    gdown.download(url, output, quiet=False)
else:
    print(f"El archivo {output} ya existe. Se omitió la descarga.")

# Cargar el archivo .pkl
inference_cache = pd.read_pickle(output)

# Verificar el tipo de objeto cargado
print(f"Tipo de objeto cargado: {type(inference_cache)}")

# %%
# Si es un diccionario, inspeccionar las claves y algunos valores
if isinstance(inference_cache, dict):
    print(f"El diccionario tiene {len(inference_cache)} claves.")


# %%


def extract_documents(inference_cache, query):
    if not isinstance(inference_cache, dict):
        raise TypeError("inference_cache must be a dictionary")
    if not isinstance(query, str):
        raise TypeError("query must be a string")

    documents = set()
    pattern = rf'''Given a query "{query}", which of the following two passages is more relevant to the query\?

Passage A: "(.*?)"

Passage B: "(.*?)"

Output Passage A or Passage B:'''

    print(f"Using pattern: {pattern}")

    for key in inference_cache.keys():
        if isinstance(key, tuple) and len(key) == 2 and isinstance(key[1], str):
            matches = re.findall(pattern, key[1], re.DOTALL)
            if matches:
                documents.update(matches[0])  # matches[0] is a tuple of (Passage A, Passage B)

    return list(documents)

# Example usage:
query = "how do they do open heart surgery"

# docs = extract_documents(inference_cache, query)

# docs[0]

# %%
query=""

# %% [markdown]
# We define a batch_compare function that simulates the process of batch inference.

# %%
from abc import ABC, abstractmethod

class SortingAlgorithm(ABC):
    @abstractmethod
    def sort(self, data, k=None):
        pass

class BaseComparator(ABC):
    def __init__(self):
        self.count = 0
        self.batch_lengths = []

    @abstractmethod
    def compare_batch(self, items, pivot):
        pass

    @abstractmethod
    def compare(self, item1, item2):
        pass

    def reset(self):
        self.count = 0
        self.batch_lengths = []

class NumericComparator(BaseComparator):
    def compare_batch(self, items, pivot):
        results = []
        for item in items:
            result = self.compare(item, pivot)
            results.append(result)
        self.batch_lengths.append(len(items))
        self.count += len(items)
        return results

    def compare(self, item1, item2):
        return item1 > item2

class DocumentComparator(BaseComparator):
    def __init__(self, model, inference_cache, key='text'):
        super().__init__()
        self.model = model
        self.inference_cache = inference_cache
        self.key = key
        self.error_keys = set()
        self.query = ""  # Initialize query attribute

    def compare_batch(self, docs, pivot):
        results = []
        for doc in docs:
            try:
                result = self.compare(doc, pivot)
                results.append(result)
            except KeyError as e:
                self.error_keys.add(str(e))
                results.append(False)
            except Exception as e:
                print(f"Unexpected error in compare_batch: {e}")
                results.append(False)
        self.batch_lengths.append(len(docs))
        self.count += len(docs)
        return results

    def compare(self, doc1, doc2):
        # Ensure doc1 and doc2 are SearchResult objects
        if not isinstance(doc1, SearchResult) or not isinstance(doc2, SearchResult):
            raise TypeError("Both arguments must be SearchResult objects")

        # Extract the text from the SearchResult objects
        text1 = getattr(doc1, self.key)
        text2 = getattr(doc2, self.key)

        compare_key = (self.model, f'Given a query "{self.query}", which of the following two passages is more relevant to the query?\n\nPassage A: "{text1}"\n\nPassage B: "{text2}"\n\nOutput Passage A or Passage B:')
        
        try:
            return self.inference_cache[compare_key] == "Passage A"
        except KeyError:
            self.error_keys.add(str(compare_key))
            raise

    def print_error_keys(self):
        print("Keys that caused KeyErrors:")
        for key in self.error_keys:
            print(key)

class QuickSort(SortingAlgorithm):
    def __init__(self, comparator):
        self.comparator = comparator

    def sort(self, data, k=None):
        if k is None:
            self.quick_sort(data, 0, len(data) - 1)
            return data
        else:
            return self.quickselect(data, 0, len(data) - 1, k-1)

    def quick_sort(self, data, low, high):
        if low < high:
            pi = self.partition(data, low, high)
            self.quick_sort(data, low, pi - 1)
            self.quick_sort(data, pi + 1, high)

    def partition(self, data, low, high):
        pivot = data[high]
        all_data = data[low:high]
        results = self.comparator.compare_batch(all_data, pivot)
        i = low
        for j in range(low, high):
            if results[j-low]:
                data[i], data[j] = data[j], data[i]
                i += 1
        data[i], data[high] = data[high], data[i]
        return i

    def quickselect(self, data, low, high, index):
        if low == high:
            return data[:index+1]
        pivot_index = self.partition(data, low, high)
        if index == pivot_index:
            return data[:pivot_index+1]
        elif index < pivot_index:
            return self.quickselect(data, low, pivot_index - 1, index)
        else:
            return self.quickselect(data, pivot_index + 1, high, index)

class InsertionSort(SortingAlgorithm):
    def __init__(self, comparator):
        self.comparator = comparator

    def sort(self, data, k=None):
        n = len(data)
        for i in range(1, n):
            key = data[i]
            j = i - 1
            while j >= 0 and self.comparator.compare(key, data[j]):
                data[j + 1] = data[j]
                j -= 1
            data[j + 1] = key
        return data[:k] if k is not None else data

class QuickInsertionMixedSort(SortingAlgorithm):
    def __init__(self, comparator):
        self.comparator = comparator
        self.quicksort = QuickSort(comparator)
        self.insertionsort = InsertionSort(comparator)

    def sort(self, data, k=None):
        if k is not None:
            top_k_data = self.quicksort.quickselect(data, 0, len(data) - 1, k-1)
            sorted_top_k_data = self.insertionsort.sort(top_k_data)
            return sorted_top_k_data
        else:
            return self.quicksort.sort(data)


# %%
# # Sample list of numbers to sort
# numbers = [42, 15, 8, 23, 16, 4, 42, 15, 37, 19, 99, 1]

# %%
import ir_datasets
from pyserini.search.lucene import LuceneSearcher
from pyserini.search import get_topics
import json
from tqdm import tqdm
from typing import List, Tuple
from transformers import T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-large')

import json
from typing import List, Tuple
import os
from tqdm import tqdm
import ir_datasets
from pyserini.search import LuceneSearcher
from pyserini.search import get_topics
import codecs

import json
from typing import List, Tuple
import os
from tqdm import tqdm
import ir_datasets
from pyserini.search import LuceneSearcher
from pyserini.search import get_topics

class SearchResult:
    def __init__(self, docid: str, score: float, text: str):
        self.docid = docid
        self.score = score
        self.text = text

class Rank:
    def __init__(self, sorting_algorithm: 'SortingAlgorithm', config: dict):
        self.sorting_algorithm = sorting_algorithm
        self.comparator = sorting_algorithm.comparator
        self.index_dir = config.get('index_dir', 'msmarco-v1-passage')
        self.topics = config.get('topics', 'dl19-passage')
        self.input_run = config.get('input_run', 'run.msmarco-v1-passage.bm25-default.dl19.txt')
        self.output_file = config.get('output_file', 'run.pairwise.txt')
        self.hits = config.get('hits', 100)
        self.query_length = config.get('query_length', 64)
        self.passage_length = config.get('passage_length', 128)
        self.ir_dataset_name = config.get('ir_dataset_name')
        self.pyserini_index = config.get('pyserini_index')

        self.query_map = {}
        self.docstore = None
        self.load_queries_and_docstore()
        self.evaluation_scores = defaultdict(dict)

    def truncate(self, text, length):
        return tokenizer.convert_tokens_to_string(tokenizer.tokenize(text)[:length])

    def load_queries_and_docstore(self):
        if self.ir_dataset_name is not None:
            try:
                dataset = ir_datasets.load(self.ir_dataset_name)
                for query in dataset.queries_iter():
                    qid = query.query_id
                    text = query.text
                    self.query_map[qid] = self.truncate(text, self.query_length)
                dataset = ir_datasets.load(self.ir_dataset_name)
                self.docstore = dataset.docs_store()
            except Exception as e:
                print(f"Error loading dataset {self.ir_dataset_name}: {e}")
                print("Available datasets:", ir_datasets.registry.keys())
                raise
        elif self.pyserini_index is not None:
            try:
                self.docstore = LuceneSearcher.from_prebuilt_index(self.pyserini_index)
                
                topics = get_topics(self.topics)
                for topic_id in list(topics.keys()):
                    text = topics[topic_id]['title']
                    self.query_map[str(topic_id)] = self.truncate(text, self.query_length)
                
                print(f"Loaded {len(self.query_map)} queries from topics: {self.topics}")
            except Exception as e:
                print(f"Error loading index or topics: {e}")
                from pyserini.search._base import topics_mapping
                print("Available topics:", list(topics_mapping.keys()))
                raise
        else:
            raise ValueError("Either 'ir_dataset_name' or 'pyserini_index' must be provided in the config.")

    def load_ranked_docs(self) -> List[Tuple[str, str, List[SearchResult]]]:
        cache_file = 'first_stage_rankings.json'
        if os.path.exists(cache_file):
            print(f'Loading cached rankings from {cache_file}.')
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                first_stage_rankings = []
                for item in data:
                    qid = item[0]
                    query = item[1]
                    docs = [SearchResult(**doc) for doc in item[2]]
                    first_stage_rankings.append((qid, query, docs))
                return first_stage_rankings

        print(f'Loading first stage run from {self.input_run}.')
        first_stage_rankings = []
        with open(self.input_run, 'r') as f:
            current_qid = None
            current_ranking = []
            for line in tqdm(f):
                qid, _, docid, _, score, _ = line.strip().split()
                if qid != current_qid:
                    if current_qid is not None:
                        first_stage_rankings.append((current_qid, self.query_map[current_qid], current_ranking[:self.hits]))
                    current_ranking = []
                    current_qid = qid
                if len(current_ranking) >= self.hits:
                    continue
                doc_text = self.fetch_doc_text(docid)


                current_ranking.append(SearchResult(docid=docid, score=float(score), text=doc_text))
            if current_qid:
                first_stage_rankings.append((current_qid, self.query_map[current_qid], current_ranking[:self.hits]))

        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump([[qid, query, [doc.__dict__ for doc in docs]] for qid, query, docs in first_stage_rankings], f, ensure_ascii=False)
        return first_stage_rankings

    def fetch_doc_text(self, docid):
        #if self.pyserini_index is not None:
            raw_doc = self.docstore.get(docid).text#.raw()
            if type(raw_doc)==str and 'come into play when you' in raw_doc:
                print("DOCTEXT:",raw_doc)
            if self.check_bad_char(raw_doc, "raw document"):
                return raw_doc  # Return early if bad char found
            try:
                data = json.loads(raw_doc)
                text = data.get('contents', '')
                if self.check_bad_char(text, "JSON contents"):
                    return text  # Return early if bad char found

                if 'title' in data:
                    title = data['title']
                    if self.check_bad_char(title, "title"):
                        return title + " " + text  # Return early if bad char found
                    text = f"{title} {text}"

            except json.JSONDecodeError:
                return raw_doc

            return self.truncate(text, self.passage_length)

    def check_bad_char(self, text, stage):
        if 'â' in text:
            print(f"Found 'â' in {stage}")
            return True
        return False        
    
    def rerank_docs(self, ranked_docs: List[Tuple[str, str, List[SearchResult]]], evaluate = False) -> None:
        reranked_results = {}
        for qid, query, docs in tqdm(ranked_docs):
            self.comparator.query = query
            reranked_docs = self.sorting_algorithm.sort(docs, self.hits)
            reranked_results[qid] = reranked_docs
        with open(self.output_file, 'w') as f:
            for qid, docs in reranked_results.items():
                for rank, doc in enumerate(docs, start=1):
                    score = 1.0 / rank
                    f.write(f"{qid} Q0 {doc.docid} {rank} {score:.6f} run\n")
        if evaluate:
            self.evaluate_run()

    def run(self):
        ranked_docs = self.load_ranked_docs()
        self.rerank_docs(ranked_docs)
        print(f"Reranking complete. Results written to {self.output_file}")

    def get_query_text(self, qid: str) -> str:
        """Get the query text for a given query ID."""
        return self.query_map.get(qid, f"Query text for {qid}")
    
    def evaluate_run(self):
        import ir_datasets
        
        script_path = download_evaluation_script('trec_eval')
        java_home = "/usr/lib/jvm/java-21-openjdk-amd64"
        java_bin = os.path.join(java_home, "bin", "java")

        # Get the qrels file path using ir_datasets
        dataset = ir_datasets.load("msmarco-passage/trec-dl-2020")
        qrels_file = dataset.qrels_path()

        cmd = [
            java_bin,
            "-jar", script_path,
            "-c",
            "-m", "ndcg_cut.10",
            qrels_file,
            self.output_file
        ]

        env = os.environ.copy()
        env["JAVA_HOME"] = java_home
        env["PATH"] = f"{os.path.join(java_home, 'bin')}:{env['PATH']}"

        print(f'Running command: {cmd}')
        try:
            process = subprocess.Popen(cmd,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    env=env)
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                print(f"Error running evaluation command. Return code: {process.returncode}")
                print(f"Error output: {stderr.decode('utf-8')}")
            else:
                print('Results:')
                result_output = stdout.decode("utf-8").rstrip()
                print(result_output)

                self.evaluation_scores.clear()
                for line in result_output.split('\n'):
                    parts = line.split()
                    if len(parts) == 3:
                        metric, scope, value = parts
                        self.evaluation_scores[metric] = float(value)
        except Exception as e:
            print(f"An error occurred while running the evaluation: {str(e)}")

        if not self.evaluation_scores:
            print("Warning: No evaluation scores were recorded.")
# %%
config = {
    'input_run': 'run.msmarco-v1-passage.bm25-default.dl20.txt',
    'output_file': 'run.pairwise.quicksort_mix.txt',
    'model_name': 'google/flan-t5-large',
    'device': 'auto',
    'topics':'dl20',
    'input_run':'run.msmarco-v1-passage.bm25-default.dl20.txt',
    'hits': 100,
    'query_length': 32,
    'passage_length': 120,
    #'pyserini_index': 'msmarco-v1-passage'  
    'ir_dataset_name':'msmarco-passage/trec-dl-2020'
}

model = 'google/flan-t5-large'
comparator = DocumentComparator(model, inference_cache)
sorting_algorithm = QuickInsertionMixedSort(comparator)

ranker = Rank(sorting_algorithm, config)
ranker.run()
ranker.evaluate_run()
# %%
# def find_keys_with_passage(inference_cache, passage_text):
#     matching_keys = []
#     for key in inference_cache.keys():
#         # Ensure the key is a tuple and has at least two elements
#         if isinstance(key, tuple) and len(key) >= 2:
#             # Check if the passage text is part of the second element of the tuple
#             if passage_text in key[1]:
#                 matching_keys.append(key)
#     return matching_keys
# # Define the passage text you're looking for
# passage_text = 'Several factors come into play when'

# # Call the function
# matching_keys = find_keys_with_passage(inference_cache, passage_text)

# # Print the matching keys
# print("Matching keys:", matching_keys)


# %% [markdown]
# We define our quickselect+insertionsort algorithm. A common way to return sorted top k by leveraging that quickselect works well on big arrays with paralellization and nearly sortedeness, and insertion sort works well with small arrays and nearly sortedeness.

# %% [markdown]
# 


