import math

class MinHeap:
    def __init__(self):
        self.heap = []

    def _heapify_up(self, index):
        parent_index = (index - 1) // 2
        if parent_index >= 0 and self.heap[parent_index] > self.heap[index]:
            self.heap[parent_index], self.heap[index] = self.heap[index], self.heap[parent_index]
            self._heapify_up(parent_index)

    def _heapify_down(self, index):
        smallest = index
        left_child_index = 2 * index + 1
        right_child_index = 2 * index + 2

        if left_child_index < len(self.heap) and self.heap[left_child_index] < self.heap[smallest]:
            smallest = left_child_index
        if right_child_index < len(self.heap) and self.heap[right_child_index] < self.heap[smallest]:
            smallest = right_child_index

        if smallest != index:
            self.heap[smallest], self.heap[index] = self.heap[index], self.heap[smallest]
            self._heapify_down(smallest)

    def insert(self, key):
        self.heap.append(key)
        self._heapify_up(len(self.heap) - 1)

    def extract_min(self):
        if len(self.heap) == 0:
            raise IndexError("Heap is empty")
        if len(self.heap) == 1:
            return self.heap.pop()

        root = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._heapify_down(0)
        return root

    def get_min(self):
        if len(self.heap) == 0:
            raise IndexError("Heap is empty")
        return self.heap[0]

    def size(self):
        return len(self.heap)

    def is_empty(self):
        return len(self.heap) == 0

    def clear(self):
        self.heap = []

    def to_list(self):
        return self.heap.copy()

    def from_list(self, lst):
        self.heap = lst.copy()
        for i in range(len(self.heap) // 2, -1, -1):
            self._heapify_down(i)

    def __repr__(self):
        return f"MinHeap({self.heap})"

    def __len__(self):
        return self.size()

    def __contains__(self, item):
        return item in self.heap

    def __iter__(self):
        return iter(self.heap)

    def __getitem__(self, index):
        return self.heap[index]

    def __setitem__(self, index, value):
        self.heap[index] = value
        self._heapify_down(index)
        self._heapify_up(index)

    def merge(self, other_heap):
        for item in other_heap:
            self.insert(item)

    def build_heap(self, iterable):
        self.heap = list(iterable)
        for i in range(len(self.heap) // 2, -1, -1):
            self._heapify_down(i)

    def increase_key(self, index, value):
        if index < 0 or index >= len(self.heap):
            raise IndexError("Index out of range")
        if value < self.heap[index]:
            raise ValueError("New value is smaller than the current value")
        self.heap[index] = value
        self._heapify_down(index)

    def decrease_key(self, index, value):
        if index < 0 or index >= len(self.heap):
            raise IndexError("Index out of range")
        if value > self.heap[index]:
            raise ValueError("New value is greater than the current value")
        self.heap[index] = value
        self._heapify_up(index)

    def remove(self, index):
        if index < 0 or index >= len(self.heap):
            raise IndexError("Index out of range")
        self.heap[index] = self.heap[-1]
        self.heap.pop()
        self._heapify_down(index)
        self._heapify_up(index)

    def replace(self, index, value):
        if index < 0 or index >= len(self.heap):
            raise IndexError("Index out of range")
        old_value = self.heap[index]
        self.heap[index] = value
        if value < old_value:
            self._heapify_up(index)
        else:
            self._heapify_down(index)
        return old_value

    def find(self, value):
        for index, item in enumerate(self.heap):
            if item == value:
                return index
        return -1

    def count(self, value):
        return self.heap.count(value)

    def level_order_traversal(self):
        return self.heap

    def preorder_traversal(self, index=0):
        if index >= len(self.heap):
            return []
        return [self.heap[index]] + self.preorder_traversal(2 * index + 1) + self.preorder_traversal(2 * index + 2)

    def inorder_traversal(self, index=0):
        if index >= len(self.heap):
            return []
        return self.inorder_traversal(2 * index + 1) + [self.heap[index]] + self.inorder_traversal(2 * index + 2)

    def postorder_traversal(self, index=0):
        if index >= len(self.heap):
            return []
        return self.postorder_traversal(2 * index + 1) + self.postorder_traversal(2 * index + 2) + [self.heap[index]]

    def find_kth_smallest(self, k):
        if k < 1 or k > len(self.heap):
            raise IndexError("Index out of range")
        return sorted(self.heap)[k - 1]

    def find_kth_largest(self, k):
        if k < 1 or k > len(self.heap):
            raise IndexError("Index out of range")
        return sorted(self.heap, reverse=True)[k - 1]

    def heapify(self):
        for i in range(len(self.heap) // 2, -1, -1):
            self._heapify_down(i)

    def heap_sort(self):
        sorted_list = []
        original_heap = self.heap.copy()
        while self.heap:
            sorted_list.append(self.extract_min())
        self.heap = original_heap
        return sorted_list

    def merge_heaps(self, other_heap):
        self.heap += other_heap
        self.heapify()

    def change_value(self, index, new_value):
        if index < 0 or index >= len(self.heap):
            raise IndexError("Index out of range")
        old_value = self.heap[index]
        self.heap[index] = new_value
        if new_value < old_value:
            self._heapify_up(index)
        else:
            self._heapify_down(index)

    def remove_value(self, value):
        index = self.find(value)
        if index == -1:
            raise ValueError("Value not found in heap")
        self.remove(index)

    def extract_all(self):
        sorted_list = []
        while self.heap:
            sorted_list.append(self.extract_min())
        return sorted_list

    def replace_root(self, value):
        if len(self.heap) == 0:
            raise IndexError("Heap is empty")
        root = self.heap[0]
        self.heap[0] = value
        self._heapify_down(0)
        return root

    def __eq__(self, other):
        if isinstance(other, MinHeap):
            return self.heap == other.heap
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if isinstance(other, MinHeap):
            return self.heap < other.heap
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, MinHeap):
            return self.heap <= other.heap
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, MinHeap):
            return self.heap > other.heap
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, MinHeap):
            return self.heap >= other.heap
        return NotImplemented

    def __call__(self, iterable):
        self.build_heap(iterable)

    def __add__(self, other):
        if isinstance(other, MinHeap):
            new_heap = MinHeap()
            new_heap.heap = self.heap + other.heap
            new_heap.heapify()
            return new_heap
        return NotImplemented

    def __iadd__(self, other):
        if isinstance(other, MinHeap):
            self.heap += other.heap
            self.heapify()
            return self
        return NotImplemented

    def __sub__(self, value):
        if isinstance(value, int):
            new_heap = MinHeap()
            new_heap.heap = [item - value for item in self.heap]
            new_heap.heapify()
            return new_heap
        return NotImplemented

    def __isub__(self, value):
        if isinstance(value, int):
            self.heap = [item - value for item in self.heap]
            self.heapify()
            return self
        return NotImplemented

    def __mul__(self, value):
        if isinstance(value, int):
            new_heap = MinHeap()
            new_heap.heap = [item * value for item in self.heap]
            new_heap.heapify()
            return new_heap
        return NotImplemented

    def __imul__(self, value):
        if isinstance(value, int):
            self.heap = [item * value for item in self.heap]
            self.heapify()
            return self
        return NotImplemented

    def __truediv__(self, value):
        if isinstance(value, int):
            new_heap = MinHeap()
            new_heap.heap = [item / value for item in self.heap]
            new_heap.heapify()
            return new_heap
        return NotImplemented

    def __itruediv__(self, value):
        if isinstance(value, int):
            self.heap = [item / value for item in self.heap]
            self.heapify()
            return self
        return NotImplemented

    def __floordiv__(self, value):
        if isinstance(value, int):
            new_heap = MinHeap()
            new_heap.heap = [item // value for item in self.heap]
            new_heap.heapify()
            return new_heap
        return NotImplemented

    def __ifloordiv__(self, value):
        if isinstance(value, int):
            self.heap = [item // value for item in self.heap]
            self.heapify()
            return self
        return NotImplemented

    def __mod__(self, value):
        if isinstance(value, int):
            new_heap = MinHeap()
            new_heap.heap = [item % value for item in self.heap]
            new_heap.heapify()
            return new_heap
        return NotImplemented

    def __imod__(self, value):
        if isinstance(value, int):
            self.heap = [item % value for item in self.heap]
            self.heapify()
            return self
        return NotImplemented

    def __pow__(self, value):
        if isinstance(value, int):
            new_heap = MinHeap()
            new_heap.heap = [item ** value for item in self.heap]
            new_heap.heapify()
            return new_heap
        return NotImplemented

    def __ipow__(self, value):
        if isinstance(value, int):
            self.heap = [item ** value for item in self.heap]
            self.heapify()
            return self
        return NotImplemented

    def __neg__(self):
        new_heap = MinHeap()
        new_heap.heap = [-item for item in self.heap]
        new_heap.heapify()
        return new_heap

    def __pos__(self):
        return self

    def __abs__(self):
        new_heap = MinHeap()
        new_heap.heap = [abs(item) for item in self.heap]
        new_heap.heapify()
        return new_heap

    def __invert__(self):
        new_heap = MinHeap()
        new_heap.heap = [~item for item in self.heap]
        new_heap.heapify()
        return new_heap

    def __and__(self, value):
        if isinstance(value, int):
            new_heap = MinHeap()
            new_heap.heap = [item & value for item in self.heap]
            new_heap.heapify()
            return new_heap
        return NotImplemented

    def __iand__(self, value):
        if isinstance(value, int):
            self.heap = [item & value for item in self.heap]
            self.heapify()
            return self
        return NotImplemented

    def __or__(self, value):
        if isinstance(value, int):
            new_heap = MinHeap()
            new_heap.heap = [item | value for item in self.heap]
            new_heap.heapify()
            return new_heap
        return NotImplemented

    def __ior__(self, value):
        if isinstance(value, int):
            self.heap = [item | value for item in self.heap]
            self.heapify()
            return self
        return NotImplemented

    def __xor__(self, value):
        if isinstance(value, int):
            new_heap = MinHeap()
            new_heap.heap = [item ^ value for item in self.heap]
            new_heap.heapify()
            return new_heap
        return NotImplemented

    def __ixor__(self, value):
        if isinstance(value, int):
            self.heap = [item ^ value for item in self.heap]
            self.heapify()
            return self
        return NotImplemented

    def __lshift__(self, value):
        if isinstance(value, int):
            new_heap = MinHeap()
            new_heap.heap = [item << value for item in self.heap]
            new_heap.heapify()
            return new_heap
        return NotImplemented

    def __ilshift__(self, value):
        if isinstance(value, int):
            self.heap = [item << value for item in self.heap]
            self.heapify()
            return self
        return NotImplemented

    def __rshift__(self, value):
        if isinstance(value, int):
            new_heap = MinHeap()
            new_heap.heap = [item >> value for item in self.heap]
            new_heap.heapify()
            return new_heap
        return NotImplemented

    def __irshift__(self, value):
        if isinstance(value, int):
            self.heap = [item >> value for item in self.heap]
            self.heapify()
            return self
        return NotImplemented

    def __round__(self):
        new_heap = MinHeap()
        new_heap.heap = [round(item) for item in self.heap]
        new_heap.heapify()
        return new_heap

    def __floor__(self):
        new_heap = MinHeap()
        new_heap.heap = [math.floor(item) for item in self.heap]
        new_heap.heapify()
        return new_heap

    def __ceil__(self):
        new_heap = MinHeap()
        new_heap.heap = [math.ceil(item) for item in self.heap]
        new_heap.heapify()
        return new_heap

    def __trunc__(self):
        new_heap = MinHeap()
        new_heap.heap = [math.trunc(item) for item in self.heap]
        new_heap.heapify()
        return new_heap

    def __matmul__(self, other):
        if isinstance(other, MinHeap):
            new_heap = MinHeap()
            new_heap.heap = [x * y for x, y in zip(self.heap, other.heap)]
            new_heap.heapify()
            return new_heap
        return NotImplemented

    def __imatmul__(self, other):
        if isinstance(other, MinHeap):
            self.heap = [x * y for x, y in zip(self.heap, other.heap)]
            self.heapify()
            return self
        return NotImplemented

    def mean(self):
        return sum(self.heap) / len(self.heap)

    def median(self):
        sorted_heap = sorted(self.heap)
        n = len(sorted_heap)
        if n % 2 == 1:
            return sorted_heap[n // 2]
        else:
            return (sorted_heap[n // 2 - 1] + sorted_heap[n // 2]) / 2

    def mode(self):
        return max(set(self.heap), key=self.heap.count)

    def variance(self):
        mean = self.mean()
        return sum((x - mean) ** 2 for x in self.heap) / len(self.heap)

    def std_deviation(self):
        return self.variance() ** 0.5

    def percentile(self, percent):
        if not 0 <= percent <= 100:
            raise ValueError("Percentile must be between 0 and 100")
        sorted_heap = sorted(self.heap)
        k = (len(sorted_heap) - 1) * percent / 100
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_heap[int(k)]
        d0 = sorted_heap[int(f)] * (c - k)
        d1 = sorted_heap[int(c)] * (k - f)
        return d0 + d1

    def histogram(self, bins=10):
        if not isinstance(bins, int) or bins <= 0:
            raise ValueError("Number of bins must be a positive integer")
        min_val, max_val = min(self.heap), max(self.heap)
        range_val = max_val - min_val
        bin_width = range_val / bins
        histogram = [0] * bins
        for value in self.heap:
            bin_index = min(int((value - min_val) / bin_width), bins - 1)
            histogram[bin_index] += 1
        return histogram

    def quantile(self, q):
        if not 0 <= q <= 1:
            raise ValueError("Quantile must be between 0 and 1")
        sorted_heap = sorted(self.heap)
        k = (len(sorted_heap) - 1) * q
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_heap[int(k)]
        d0 = sorted_heap[int(f)] * (c - k)
        d1 = sorted_heap[int(c)] * (k - f)
        return d0 + d1

    def skewness(self):
        mean = self.mean()
        std_dev = self.std_deviation()
        return sum(((x - mean) / std_dev) ** 3 for x in self.heap) / len(self.heap)

    def kurtosis(self):
        mean = self.mean()
        std_dev = self.std_deviation()
        return sum(((x - mean) / std_dev) ** 4 for x in self.heap) / len(self.heap) - 3

    def covariance(self, other_heap):
        if not isinstance(other_heap, MinHeap):
            raise TypeError("Argument must be a MinHeap")
        if len(self.heap) != len(other_heap.heap):
            raise ValueError("Heaps must have the same length")
        
        mean_self = self.mean()
        mean_other = other_heap.mean()
        
        cov = sum((x - mean_self) * (y - mean_other) for x, y in zip(self.heap, other_heap.heap)) / len(self.heap)
        
        return cov
