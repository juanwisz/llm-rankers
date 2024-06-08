from llmrankers.rankers import LlmRanker, SearchResult
from llmrankers.pairwise import PairwiseLlmRanker, Text2TextGenerationDataset
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorWithPadding
from typing import List
import copy
import torch
# We introduce QKV-Sort, Quick-K-Vectorized Sort. A sorting algorithm optimized to work on reranking tasks using a large pre-trained transformers for pairwise comparisons by neural inference, we make it both latency and cost efficient by leveraging the pre-rank, a potential cache and batch inference usage.
# We leverage (all in the same algorithm):
#1. Nearly sorted pre-ranked documents.
#2. Batch inference
#3. Only top K head of the array is required.
#4. Existance of a cache that we can leverage to spend less is tipically at hand. (not implemented yet)
# Best-Case complexity is 1*n/b inferences (yes, exactly n without batch, much less than b if we have a batch).
# Avg.-Case complexity is below k*n even without batching, so it's k*n/b (needs more detailed calculation and an experiment over ir_datasets and BEIR datasets, pre-ranking on BM25).
# Worst Case complexity is still (n-0)/b+(n-1)/b+(n-2)/b+...+(n-k)/b = O(n^2/b) (highly unlikely)(needs more detailed calculation).
# If cached comparisons are available, we maximize it's usage while minimizing uncached comparisons, and complexity drops drops potentially even more.
class Quicksort(PairwiseLlmRanker):
    def __init__(self, *args, **kwargs):
        self.passage_length = kwargs.pop('passage_length', 256)
        super().__init__(*args, **kwargs)

    def truncate(self, text, length):
        return self.tokenizer.convert_tokens_to_string(self.tokenizer.tokenize(text)[:length])
    
    def partition(self, docs, pivot):
        less = []
        greater = []
        pivot_doc = docs[pivot]
        batch_inputs = []
        batch_index_to_doc_index = {}
        batch_index = 0

        truncated_pivot_doc = self.truncate(pivot_doc.text, self.passage_length)
        
        for doc_index, doc in enumerate(docs):
            if doc != pivot_doc:
                truncated_doc = self.truncate(doc.text, self.passage_length)
                batch_index_to_doc_index[batch_index] = doc_index
                batch_inputs.append(self.prompt.format(query=self.query, doc1=truncated_doc, doc2=truncated_pivot_doc))
                batch_index += 1
                batch_index_to_doc_index[batch_index] = doc_index
                batch_inputs.append(self.prompt.format(query=self.query, doc1=truncated_pivot_doc, doc2=truncated_doc))
                batch_index += 1
        
        dataset = Text2TextGenerationDataset(batch_inputs, self.tokenizer)
        loader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=DataCollatorWithPadding(self.tokenizer))
        outputs = []
        
        for batch in loader:

            self.total_inference += 1
            self.total_compare += len(batch['input_ids'])
            input_ids = batch['input_ids'].to(self.device)

            current_batch_size = input_ids.shape[0]
            current_decoder_input_ids = self.decoder_input_ids[:current_batch_size]
            print(f"Input Shape 0,1: {(input_ids.shape[0],  input_ids.shape[1])}")
            if input_ids.shape[1]>300:
                print('hola')
            output_ids = self.llm.generate(
                input_ids,
                decoder_input_ids=current_decoder_input_ids,
                max_new_tokens=2
            )

            print(f"Output Shape 0,1: {(output_ids.shape[0],  output_ids.shape[1])}")
            outputs.extend(self.tokenizer.batch_decode(output_ids, skip_special_tokens=True))

            self.total_prompt_tokens += input_ids.shape[0] * input_ids.shape[1]
            self.total_completion_tokens += output_ids.shape[0] * output_ids.shape[1]


            del input_ids, current_decoder_input_ids, output_ids
            torch.cuda.empty_cache()
            print(f"Finished partition call with amount of docs {len(docs)}")

            

        for batch_index in range(0, len(outputs), 2):
            doc_index = batch_index_to_doc_index[batch_index]
            if outputs[batch_index] == "Passage A" and outputs[batch_index + 1] == "Passage B":
                less.append(docs[doc_index])
            else:
                greater.append(docs[doc_index])
        
        return less, greater

    def quickselect(self, docs, k):
        if len(docs) == k:
            return self.quicksort(docs)

        pivot_index = k  # Choosing k as pivot index for top k elements
        less, greater = self.partition(docs, pivot_index)
        
        # Calculate the length of the 'less' partition
        len_less = len(less)
        
        # Determine which partition to recurse into
        if k <= len_less:
            return self.quickselect(less, k)
        elif k == len_less + 1: #plus pivot
            # If k is exactly the length of 'less', return 'less' plus the pivot
            return self.quicksort(less + [docs[pivot_index]])
        elif k > len_less + 1: #plus pivot
            # If k is greater than the length of 'less', recurse into 'greater'
            # Adjust k to be relative to the size of 'greater'
            return self.quicksort(less + [docs[pivot_index]] + self.quickselect(greater, k - len_less - 1))
        else:
            # This condition should not normally be reached, but it's here for completeness
            raise ValueError("Something is wrong")

    def quicksort(self, docs):
        if len(docs) < 2:
            return docs
        pivot_index = len(docs) // 2
        less, greater = self.partition(docs, pivot_index)
        return self.quicksort(less) + [docs[pivot_index]] + self.quicksort(greater)

    def rerank(self, query: str, ranking: List[SearchResult]) -> List[SearchResult]:
        original_ranking = copy.deepcopy(ranking)
        self.total_compare = 0
        self.total_inference = 0
        # self.total_cache_hits = 0  # Uncomment and implement if cache logic is added later
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0
        self.query = query
        k = min(self.k, len(ranking))
        top_docs = self.quickselect(ranking, k)  # Directly use 'k' to denote the number of top documents required
        results = []
        top_doc_ids = set()
        rank = 1
        for doc in top_docs:
            top_doc_ids.add(doc.docid)
            results.append(SearchResult(docid=doc.docid, score=-rank, text=None))
            rank += 1
        for doc in original_ranking:
            if doc.docid not in top_doc_ids:
                results.append(SearchResult(docid=doc.docid, score=-rank, text=None))
                rank += 1
        return results
