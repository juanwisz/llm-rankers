import pytest
import random
from llmrankers.run_ours import QuickSort  # Assuming the QuickSort class is in a file named quicksort.py

class SimpleComparator:
    def compare(self, a, b):
        return a < b

    def compare_batch(self, data, pivot):
        return [self.compare(item, pivot) for item in data]

def generate_test_arrays(n, num_arrays):
    """Generate a variety of test arrays."""
    arrays = []
    for _ in range(num_arrays):
        array_type = random.choice([
            'random',
            'sorted',
            'reverse_sorted',
            'nearly_sorted',
            'few_unique',
            'many_duplicates'
        ])
        
        if array_type == 'random':
            arrays.append([random.randint(0, n*10) for _ in range(n)])
        elif array_type == 'sorted':
            arrays.append(list(range(n)))
        elif array_type == 'reverse_sorted':
            arrays.append(list(range(n-1, -1, -1)))
        elif array_type == 'nearly_sorted':
            arr = list(range(n))
            for _ in range(n // 10):
                i, j = random.sample(range(n), 2)
                arr[i], arr[j] = arr[j], arr[i]
            arrays.append(arr)
        elif array_type == 'few_unique':
            arrays.append([random.randint(0, n//10) for _ in range(n)])
        elif array_type == 'many_duplicates':
            arr = [random.randint(0, n//2) for _ in range(n//2)]
            arr.extend(random.choices(arr, k=n-len(arr)))
            random.shuffle(arr)
            arrays.append(arr)
    
    return arrays

@pytest.fixture(params=['original', 'middle', 'random', 'median_of_three'])
def quicksort_strategy(request):
    return QuickSort(SimpleComparator(), pivot_strategy=request.param)

@pytest.mark.parametrize("array_length", [10, 100, 1000])
@pytest.mark.parametrize("input_array", generate_test_arrays(10, 5))  # 5 arrays for each length
def test_quicksort_strategies(quicksort_strategy, array_length, input_array):
    # Scale the input array to the desired length
    scaled_array = [x * (array_length // len(input_array)) for x in input_array]
    while len(scaled_array) < array_length:
        scaled_array.append(random.choice(scaled_array))
    random.shuffle(scaled_array)
    
    # Make a copy of the input array to avoid modifying the original
    array_copy = scaled_array.copy()
    
    # Sort using the current strategy
    sorted_array = quicksort_strategy.sort(array_copy)
    
    # Check if the array is correctly sorted
    assert sorted_array == sorted(scaled_array), f"Failed to sort array of length {array_length} using {quicksort_strategy.pivot_strategy.__name__}"

    # Check if the original array was modified in-place
    assert array_copy == sorted_array, f"Array was not sorted in-place using {quicksort_strategy.pivot_strategy.__name__}"

def test_quickselect():
    qs = QuickSort(SimpleComparator())
    for array_length in [10, 100, 1000]:
        for input_array in generate_test_arrays(array_length, 2):
            for k in [1, len(input_array) // 2, len(input_array)]:
                result = qs.sort(input_array.copy(), k)
                expected = sorted(input_array)[:k]
                assert result == expected, f"Quickselect failed for k={k} on array of length {len(input_array)}"

if __name__ == "__main__":
    pytest.main([__file__])