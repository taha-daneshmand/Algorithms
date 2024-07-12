class Array:
    def __init__(self, capacity=1, fill_value=None):
        self.capacity = max(capacity, 1)
        self.count = 0
        self.items = [fill_value] * self.capacity

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if 0 <= index < self.count:
            return self.items[index]
        raise IndexError("Array index out of range")

    def __setitem__(self, index, value):
        if 0 <= index < self.count:
            self.items[index] = value
        else:
            raise IndexError("Array index out of range")

    def append(self, item):
        if self.count == self.capacity:
            self._resize(2 * self.capacity)
        self.items[self.count] = item
        self.count += 1

    def insert(self, index, item):
        if 0 <= index <= self.count:
            if self.count == self.capacity:
                self._resize(2 * self.capacity)
            for i in range(self.count, index, -1):
                self.items[i] = self.items[i-1]
            self.items[index] = item
            self.count += 1
        else:
            raise IndexError("Array index out of range")

    def remove(self, item):
        for i in range(self.count):
            if self.items[i] == item:
                self._delete(i)
                return
        raise ValueError("Item not found in the array")

    def pop(self, index=-1):
        if index == -1:
            index = self.count - 1
        if 0 <= index < self.count:
            item = self.items[index]
            self._delete(index)
            return item
        raise IndexError("Array index out of range")

    def clear(self):
        self.count = 0
        self.capacity = 1
        self.items = [None]

    def index(self, item):
        for i in range(self.count):
            if self.items[i] == item:
                return i
        raise ValueError("Item not found in the array")

    def count(self, item):
        return sum(1 for i in range(self.count) if self.items[i] == item)

    def reverse(self):
        for i in range(self.count // 2):
            self.items[i], self.items[self.count-1-i] = self.items[self.count-1-i], self.items[i]

    def extend(self, iterable):
        for item in iterable:
            self.append(item)

    def _delete(self, index):
        for i in range(index, self.count - 1):
            self.items[i] = self.items[i + 1]
        self.count -= 1
        if self.count < self.capacity // 4:
            self._resize(self.capacity // 2)

    def _resize(self, new_capacity):
        new_array = [None] * new_capacity
        for i in range(self.count):
            new_array[i] = self.items[i]
        self.items = new_array
        self.capacity = new_capacity

    def __iter__(self):
        return (self.items[i] for i in range(self.count))

    def __repr__(self):
        return f"Array({', '.join(repr(self.items[i]) for i in range(self.count))})"

    def __eq__(self, other):
        if isinstance(other, Array):
            return self.items[:self.count] == other.items[:other.count]
        return False

    def sort(self, key=None, reverse=False):
        self.items[:self.count] = sorted(self.items[:self.count], key=key, reverse=reverse)

    def binary_search(self, item):
        left, right = 0, self.count - 1
        while left <= right:
            mid = (left + right) // 2
            if self.items[mid] == item:
                return mid
            elif self.items[mid] < item:
                left = mid + 1
            else:
                right = mid - 1
        return -1  # Item not found

    def map(self, func):
        for i in range(self.count):
            self.items[i] = func(self.items[i])

    def filter(self, pred):
        new_array = Array()
        for item in self:
            if pred(item):
                new_array.append(item)
        return new_array

    def reduce(self, func, initial=None):
        if self.count == 0:
            return initial
        if initial is None:
            result = self.items[0]
            start = 1
        else:
            result = initial
            start = 0
        for i in range(start, self.count):
            result = func(result, self.items[i])
        return result
        
    def min(self):
        if self.count == 0:
            raise ValueError("Array is empty")
        return min(self.items[:self.count])

    def max(self):
        if self.count == 0:
            raise ValueError("Array is empty")
        return max(self.items[:self.count])

    def sum(self):
        return sum(self.items[:self.count])

    def product(self):
        result = 1
        for i in range(self.count):
            result *= self.items[i]
        return result

    def mean(self):
        if self.count == 0:
            raise ValueError("Array is empty")
        return self.sum() / self.count

    def median(self):
        if self.count == 0:
            raise ValueError("Array is empty")
        sorted_items = sorted(self.items[:self.count])
        mid = self.count // 2
        if self.count % 2 == 0:
            return (sorted_items[mid - 1] + sorted_items[mid]) / 2
        return sorted_items[mid]

    def mode(self):
        if self.count == 0:
            raise ValueError("Array is empty")
        frequency = {}
        for item in self.items[:self.count]:
            frequency[item] = frequency.get(item, 0) + 1
        max_count = max(frequency.values())
        modes = [k for k, v in frequency.items() if v == max_count]
        return modes if len(modes) > 1 else modes[0]

    def variance(self):
        if self.count == 0:
            raise ValueError("Array is empty")
        mean_value = self.mean()
        return sum((x - mean_value) ** 2 for x in self.items[:self.count]) / self.count

    def std_dev(self):
        return self.variance() ** 0.5

    def all(self):
        return all(self.items[:self.count])

    def any(self):
        return any(self.items[:self.count])

    def is_empty(self):
        return self.count == 0

    def is_sorted(self):
        return all(self.items[i] <= self.items[i + 1] for i in range(self.count - 1))

    def copy(self):
        new_array = Array(self.capacity)
        new_array.items = self.items[:]
        new_array.count = self.count
        return new_array

    def swap(self, i, j):
        if 0 <= i < self.count and 0 <= j < self.count:
            self.items[i], self.items[j] = self.items[j], self.items[i]
        else:
            raise IndexError("Array index out of range")

    def rotate_left(self, n):
        n = n % self.count
        self.items[:self.count] = self.items[n:self.count] + self.items[:n]

    def rotate_right(self, n):
        n = n % self.count
        self.items[:self.count] = self.items[self.count - n:self.count] + self.items[:self.count - n]

    def unique(self):
        seen = set()
        unique_items = [item for item in self.items[:self.count] if not (item in seen or seen.add(item))]
        new_array = Array()
        new_array.extend(unique_items)
        return new_array

    def intersect(self, other):
        if not isinstance(other, Array):
            raise ValueError("Argument must be of type Array")
        return Array().extend(item for item in self if item in other)

    def union(self, other):
        if not isinstance(other, Array):
            raise ValueError("Argument must be of type Array")
        new_array = self.copy()
        for item in other:
            if item not in new_array:
                new_array.append(item)
        return new_array

    def difference(self, other):
        if not isinstance(other, Array):
            raise ValueError("Argument must be of type Array")
        return Array().extend(item for item in self if item not in other)

    def symmetric_difference(self, other):
        if not isinstance(other, Array):
            raise ValueError("Argument must be of type Array")
        return Array().extend(item for item in self if item not in other).extend(item for item in other if item not in self)

    def to_list(self):
        return self.items[:self.count]

    def from_list(self, lst):
        self.clear()
        self.extend(lst)

    def find(self, predicate):
        for i in range(self.count):
            if predicate(self.items[i]):
                return self.items[i]
        return None

    def find_all(self, predicate):
        return Array().extend(item for item in self if predicate(item))

    def partition(self, predicate):
        true_array = Array()
        false_array = Array()
        for item in self:
            if predicate(item):
                true_array.append(item)
            else:
                false_array.append(item)
        return true_array, false_array

    def split(self, sep=None, maxsplit=-1):
        lst = self.to_list()
        parts = []
        current_part = []
        for item in lst:
            if item == sep:
                parts.append(Array().extend(current_part))
                current_part = []
                if maxsplit != -1:
                    maxsplit -= 1
                    if maxsplit == 0:
                        break
            else:
                current_part.append(item)
        parts.append(Array().extend(current_part))
        return parts

    def join(self, sep=""):
        return sep.join(str(item) for item in self.items[:self.count])

    def pad(self, width, fill_value=None):
        if self.count < width:
            self.extend([fill_value] * (width - self.count))

    def resize(self, new_capacity):
        self._resize(new_capacity)

    def get(self, index, default=None):
        if 0 <= index < self.count:
            return self.items[index]
        return default

    def setdefault(self, index, default=None):
        if 0 <= index < self.count:
            return self.items[index]
        else:
            self.append(default)
            return default

    def sort_desc(self):
        self.sort(reverse=True)

    def sort_asc(self):
        self.sort(reverse=False)

    def sort_key(self, key):
        self.sort(key=key)

    def swap_first_last(self):
        if self.count > 1:
            self.items[0], self.items[self.count - 1] = self.items[self.count - 1], self.items[0]

    def shuffle(self):
        import random
        random.shuffle(self.items[:self.count])

    def remove_all(self, item):
        while item in self.items[:self.count]:
            self.remove(item)

    def remove_at(self, index):
        self.pop(index)

    def replace(self, old_item, new_item):
        for i in range(self.count):
            if self.items[i] == old_item:
                self.items[i] = new_item

    def replace_all(self, old_item, new_item):
        for i in range(self.count):
            if self.items[i] == old_item:
                self.items[i] = new_item

    def trim(self):
        while self.count > 0 and self.items[self.count - 1] is None:
            self.count -= 1

    def compact(self):
        new_array = Array()
        for item in self.items[:self.count]:
            if item is not None:
                new_array.append(item)
        self.items = new_array.items
        self.count = new_array.count

    def fill(self, value):
        for i in range(self.count):
            self.items[i] = value

    def permute(self, perm):
        if len(perm) != self.count:
            raise ValueError("Permutation length must be equal to array length")
        new_array = [None] * self.count
        for i in range(self.count):
            new_array[i] = self.items[perm[i]]
        self.items = new_array

    def histogram(self):
        hist = {}
        for item in self.items[:self.count]:
            hist[item] = hist.get(item, 0) + 1
        return hist

    def cumsum(self):
        result = Array()
        total = 0
        for item in self.items[:self.count]:
            total += item
            result.append(total)
        return result

    def cumprod(self):
        result = Array()
        total = 1
        for item in self.items[:self.count]:
            total *= item
            result.append(total)
        return result

    def cummax(self):
        result = Array()
        max_val = self.items[0]
        for item in self.items[:self.count]:
            if item > max_val:
                max_val = item
            result.append(max_val)
        return result

    def cummin(self):
        result = Array()
        min_val = self.items[0]
        for item in self.items[:self.count]:
            if item < min_val:
                min_val = item
            result.append(min_val)
        return result

    def normalize(self):
        total = self.sum()
        for i in range(self.count):
            self.items[i] /= total

    def standardize(self):
        mean_val = self.mean()
        std_dev = self.std_dev()
        for i in range(self.count):
            self.items[i] = (self.items[i] - mean_val) / std_dev

    def dot(self, other):
        if not isinstance(other, Array) or self.count != other.count:
            raise ValueError("Both arrays must have the same length")
        return sum(self.items[i] * other.items[i] for i in range(self.count))

    def slice(self, start, stop, step=1):
        return Array().extend(self.items[start:stop:step])

    def unique_count(self):
        return len(set(self.items[:self.count]))

    def argmin(self):
        if self.count == 0:
            raise ValueError("Array is empty")
        return min(range(self.count), key=lambda i: self.items[i])

    def argmax(self):
        if self.count == 0:
            raise ValueError("Array is empty")
        return max(range(self.count), key=lambda i: self.items[i])

    def percentile(self, p):
        if self.count == 0:
            raise ValueError("Array is empty")
        k = (self.count - 1) * p / 100
        f = int(k)
        c = k - f
        if f + 1 < self.count:
            return self.items[f] + (self.items[f + 1] - self.items[f]) * c
        return self.items[f]

    def clip(self, min_value, max_value):
        for i in range(self.count):
            if self.items[i] < min_value:
                self.items[i] = min_value
            elif self.items[i] > max_value:
                self.items[i] = max_value

    def roll(self, n):
        n = n % self.count
        self.items[:self.count] = self.items[-n:] + self.items[:-n]

    def clip_max(self, max_value):
        for i in range(self.count):
            if self.items[i] > max_value:
                self.items[i] = max_value

    def clip_min(self, min_value):
        for i in range(self.count):
            if self.items[i] < min_value:
                self.items[i] = min_value

    def arg_sort(self):
        return sorted(range(self.count), key=lambda i: self.items[i])

    def add_prefix(self, prefix):
        return Array().extend(f"{prefix}{item}" for item in self.items[:self.count])

    def add_suffix(self, suffix):
        return Array().extend(f"{item}{suffix}" for item in self.items[:self.count])

    def to_set(self):
        return set(self.items[:self.count])

    def to_tuple(self):
        return tuple(self.items[:self.count])

    def to_dict(self, key_func):
        return {key_func(item): item for item in self.items[:self.count]}

    def from_set(self, s):
        self.clear()
        self.extend(s)

    def from_tuple(self, t):
        self.clear()
        self.extend(t)

    def from_dict(self, d, value_func=lambda x: x):
        self.clear()
        self.extend(value_func(value) for key, value in d.items())

    def sample(self, k):
        import random
        if k > self.count:
            raise ValueError("Sample larger than population")
        indices = random.sample(range(self.count), k)
        return Array().extend(self.items[i] for i in indices)

    def flatten(self):
        flat_array = Array()
        for item in self:
            if isinstance(item, Array):
                flat_array.extend(item.flatten())
            else:
                flat_array.append(item)
        return flat_array

    def unflatten(self, shape):
        if self.count != shape[0] * shape[1]:
            raise ValueError("Shape does not match array size")
        matrix = Array()
        for i in range(shape[0]):
            row = Array()
            for j in range(shape[1]):
                row.append(self.items[i * shape[1] + j])
            matrix.append(row)
        return matrix

    def chunk(self, chunk_size):
        if chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        return Array().extend(Array().extend(self.items[i:i + chunk_size]) for i in range(0, self.count, chunk_size))

    def apply(self, func):
        for i in range(self.count):
            self.items[i] = func(self.items[i])

    def transform(self, func):
        new_array = Array()
        for i in range(self.count):
            new_array.append(func(self.items[i]))
        return new_array

    def reduce_max(self):
        return self.reduce(lambda a, b: a if a > b else b)

    def reduce_min(self):
        return self.reduce(lambda a, b: a if a < b else b)

    def reduce_sum(self):
        return self.reduce(lambda a, b: a + b)

    def reduce_prod(self):
        return self.reduce(lambda a, b: a * b)
