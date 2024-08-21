from typing import List, Any, Callable, Optional, Generator

class Array:
    def __init__(self, capacity: int = 1, fill_value: Any = None):
        """
        Initializes the array with the given capacity and fills it with the fill_value.
        """
        self.capacity = max(capacity, 1)
        self.count = 0
        self.items: List[Any] = [fill_value] * self.capacity

    def __len__(self) -> int:
        """
        Returns the number of elements in the array.
        """
        return self.count

    def __getitem__(self, index: int) -> Any:
        """
        Returns the item at the given index.
        """
        if 0 <= index < self.count:
            return self.items[index]
        raise IndexError("Array index out of range")

    def __setitem__(self, index: int, value: Any):
        """
        Sets the item at the given index to the specified value.
        """
        if 0 <= index < self.count:
            self.items[index] = value
        else:
            raise IndexError("Array index out of range")

    def __contains__(self, item: Any) -> bool:
        """
        Returns True if the item is in the array, otherwise False.
        """
        return self.index(item) != -1

    def __iter__(self) -> Generator[Any, None, None]:
        """
        Returns an iterator for the array.
        """
        return (self.items[i] for i in range(self.count))

    def __repr__(self) -> str:
        """
        Returns a string representation of the array.
        """
        return f"Array({', '.join(repr(self.items[i]) for i in range(self.count))})"

    def __eq__(self, other: Any) -> bool:
        """
        Compares this array with another array.
        """
        if isinstance(other, Array):
            return self.items[:self.count] == other.items[:other.count]
        return False

    def append(self, item: Any):
        """
        Appends an item to the end of the array.
        """
        if self.count == self.capacity:
            self._resize(2 * self.capacity)
        self.items[self.count] = item
        self.count += 1

    def insert(self, index: int, item: Any):
        """
        Inserts an item at the specified index.
        """
        if 0 <= index <= self.count:
            if self.count == self.capacity:
                self._resize(2 * self.capacity)
            for i in range(self.count, index, -1):
                self.items[i] = self.items[i-1]
            self.items[index] = item
            self.count += 1
        else:
            raise IndexError("Array index out of range")

    def remove(self, item: Any):
        """
        Removes the first occurrence of the specified item from the array.
        """
        for i in range(self.count):
            if self.items[i] == item:
                self._delete(i)
                return
        raise ValueError("Item not found in the array")

    def pop(self, index: int = -1) -> Any:
        """
        Removes and returns the item at the specified index (or the last item if no index is specified).
        """
        if index == -1:
            index = self.count - 1
        if 0 <= index < self.count:
            item = self.items[index]
            self._delete(index)
            return item
        raise IndexError("Array index out of range")

    def clear(self):
        """
        Clears the array.
        """
        self.count = 0
        self.capacity = 1
        self.items = [None] * self.capacity

    def index(self, item: Any) -> int:
        """
        Returns the index of the first occurrence of the specified item.
        """
        for i in range(self.count):
            if self.items[i] == item:
                return i
        raise ValueError("Item not found in the array")

    def count(self, item: Any) -> int:
        """
        Returns the number of occurrences of the specified item.
        """
        return sum(1 for i in range(self.count) if self.items[i] == item)

    def reverse(self):
        """
        Reverses the array in place.
        """
        for i in range(self.count // 2):
            self.items[i], self.items[self.count - 1 - i] = self.items[self.count - 1 - i], self.items[i]

    def extend(self, iterable: List[Any]):
        """
        Extends the array by appending elements from the iterable.
        """
        for item in iterable:
            self.append(item)

    def _delete(self, index: int):
        """
        Deletes the item at the specified index.
        """
        for i in range(index, self.count - 1):
            self.items[i] = self.items[i + 1]
        self.count -= 1
        if self.count < self.capacity // 4:
            self._resize(max(self.capacity // 2, 1))

    def _resize(self, new_capacity: int):
        """
        Resizes the internal array to the new capacity.
        """
        new_array = [None] * new_capacity
        for i in range(self.count):
            new_array[i] = self.items[i]
        self.items = new_array
        self.capacity = new_capacity

    def sort(self, key: Optional[Callable[[Any], Any]] = None, reverse: bool = False):
        """
        Sorts the array in place.
        """
        self.items[:self.count] = sorted(self.items[:self.count], key=key, reverse=reverse)

    def binary_search(self, item: Any) -> int:
        """
        Performs binary search on the sorted array and returns the index of the item.
        """
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

    def map(self, func: Callable[[Any], Any]):
        """
        Applies the given function to each item in the array.
        """
        for i in range(self.count):
            self.items[i] = func(self.items[i])

    def filter(self, pred: Callable[[Any], bool]) -> 'Array':
        """
        Returns a new array containing only the items that satisfy the predicate.
        """
        new_array = Array()
        for item in self:
            if pred(item):
                new_array.append(item)
        return new_array

    def reduce(self, func: Callable[[Any, Any], Any], initial: Optional[Any] = None) -> Any:
        """
        Reduces the array to a single value using the given function.
        """
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

    def min(self) -> Any:
        """
        Returns the minimum value in the array.
        """
        if self.count == 0:
            raise ValueError("Array is empty")
        return min(self.items[:self.count])

    def max(self) -> Any:
        """
        Returns the maximum value in the array.
        """
        if self.count == 0:
            raise ValueError("Array is empty")
        return max(self.items[:self.count])

    def sum(self) -> Any:
        """
        Returns the sum of all the elements in the array.
        """
        return sum(self.items[:self.count])

    def product(self) -> Any:
        """
        Returns the product of all the elements in the array.
        """
        result = 1
        for i in range(self.count):
            result *= self.items[i]
        return result

    def mean(self) -> float:
        """
        Returns the mean (average) of all the elements in the array.
        """
        if self.count == 0:
            raise ValueError("Array is empty")
        return self.sum() / self.count

    def median(self) -> float:
        """
        Returns the median value of the array.
        """
        if self.count == 0:
            raise ValueError("Array is empty")
        sorted_items = sorted(self.items[:self.count])
        mid = self.count // 2
        if self.count % 2 == 0:
            return (sorted_items[mid - 1] + sorted_items[mid]) / 2
        return sorted_items[mid]

    def mode(self) -> Any:
        """
        Returns the mode (most common value) in the array.
        """
        if self.count == 0:
            raise ValueError("Array is empty")
        frequency = {}
        for item in self.items[:self.count]:
            frequency[item] = frequency.get(item, 0) + 1
        max_count = max(frequency.values())
        modes = [k for k, v in frequency.items() if v == max_count]
        return modes if len(modes) > 1 else modes[0]

    def variance(self) -> float:
        """
        Returns the variance of the elements in the array.
        """
        if self.count == 0:
            raise ValueError("Array is empty")
        mean_value = self.mean()
        return sum((x - mean_value) ** 2 for x in self.items[:self.count]) / self.count

    def std_dev(self) -> float:
        """
        Returns the standard deviation of the elements in the array.
        """
        return self.variance() ** 0.5

    def all(self) -> bool:
        """
        Returns True if all elements in the array are true, otherwise False.
        """
        return all(self.items[:self.count])

    def any(self) -> bool:
        """
        Returns True if any element in the array is true, otherwise False.
        """
        return any(self.items[:self.count])

    def is_empty(self) -> bool:
        """
        Returns True if the array is empty, otherwise False.
        """
        return self.count == 0

    def is_sorted(self) -> bool:
        """
        Returns True if the array is sorted in ascending order, otherwise False.
        """
        return all(self.items[i] <= self.items[i + 1] for i in range(self.count - 1))

    def copy(self) -> 'Array':
        """
        Returns a shallow copy of the array.
        """
        new_array = Array(self.capacity)
        new_array.items = self.items[:self.count]
        new_array.count = self.count
        return new_array

    def deepcopy(self) -> 'Array':
        """
        Returns a deep copy of the array.
        """
        new_array = Array(self.capacity)
        new_array.items = [item for item in self.items[:self.count]]
        new_array.count = self.count
        return new_array

    def swap(self, i: int, j: int):
        """
        Swaps the elements at the specified indices.
        """
        if 0 <= i < self.count and 0 <= j < self.count:
            self.items[i], self.items[j] = self.items[j], self.items[i]
        else:
            raise IndexError("Array index out of range")

    def rotate_left(self, n: int):
        """
        Rotates the array to the left by n steps.
        """
        n = n % self.count
        self.items[:self.count] = self.items[n:self.count] + self.items[:n]

    def rotate_right(self, n: int):
        """
        Rotates the array to the right by n steps.
        """
        n = n % self.count
        self.items[:self.count] = self.items[self.count - n:self.count] + self.items[:self.count - n]

    def unique(self) -> 'Array':
        """
        Returns a new array with unique elements from the original array.
        """
        seen = set()
        unique_items = [item for item in self.items[:self.count] if not (item in seen or seen.add(item))]
        new_array = Array()
        new_array.extend(unique_items)
        return new_array

    def to_list(self) -> List[Any]:
        """
        Converts the array to a list.
        """
        return self.items[:self.count]

    def from_list(self, lst: List[Any]):
        """
        Initializes the array with elements from the list.
        """
        self.clear()
        self.extend(lst)

    def to_set(self) -> set:
        """
        Converts the array to a set.
        """
        return set(self.items[:self.count])

    def to_tuple(self) -> tuple:
        """
        Converts the array to a tuple.
        """
        return tuple(self.items[:self.count])

    def trim(self):
        """
        Trims the array to its current size.
        """
        self.items = self.items[:self.count]

    def compact(self):
        """
        Removes None values from the array.
        """
        new_array = Array()
        for item in self.items[:self.count]:
            if item is not None:
                new_array.append(item)
        self.items = new_array.items
        self.count = new_array.count

    def fill(self, value: Any):
        """
        Fills the array with the specified value.
        """
        for i in range(self.count):
            self.items[i] = value

    def sample(self, k: int) -> 'Array':
        """
        Returns a new array with k random elements from the original array.
        """
        import random
        if k > self.count:
            raise ValueError("Sample larger than population")
        indices = random.sample(range(self.count), k)
        return Array().extend(self.items[i] for i in indices)

    def flatten(self) -> 'Array':
        """
        Flattens the array if it contains nested arrays.
        """
        flat_array = Array()
        for item in self:
            if isinstance(item, Array):
                flat_array.extend(item.flatten())
            else:
                flat_array.append(item)
        return flat_array

    def unflatten(self, shape: tuple) -> 'Array':
        """
        Converts a flat array into a matrix (2D array) with the given shape.
        """
        if self.count != shape[0] * shape[1]:
            raise ValueError("Shape does not match array size")
        matrix = Array()
        for i in range(shape[0]):
            row = Array()
            for j in range(shape[1]):
                row.append(self.items[i * shape[1] + j])
            matrix.append(row)
        return matrix

    def chunk(self, chunk_size: int) -> 'Array':
        """
        Splits the array into chunks of the given size.
        """
        if chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        return Array().extend(Array().extend(self.items[i:i + chunk_size]) for i in range(0, self.count, chunk_size))

    def apply(self, func: Callable[[Any], Any]):
        """
        Applies the specified function to each element in the array.
        """
        for i in range(self.count):
            self.items[i] = func(self.items[i])

    def transform(self, func: Callable[[Any], Any]) -> 'Array':
        """
        Returns a new array where each element is the result of applying the function to the corresponding element.
        """
        new_array = Array()
        for i in range(self.count):
            new_array.append(func(self.items[i]))
        return new_array

    def percentile(self, p: float) -> float:
        """
        Returns the pth percentile of the array.
        """
        if self.count == 0:
            raise ValueError("Array is empty")
        k = (self.count - 1) * p / 100
        f = int(k)
        c = k - f
        if f + 1 < self.count:
            return self.items[f] + (self.items[f + 1] - self.items[f]) * c
        return self.items[f]

    def clip(self, min_value: float, max_value: float):
        """
        Clips the array's values between min_value and max_value.
        """
        for i in range(self.count):
            if self.items[i] < min_value:
                self.items[i] = min_value
            elif self.items[i] > max_value:
                self.items[i] = max_value

    def roll(self, n: int):
        """
        Rolls the array elements by n steps.
        """
        n = n % self.count
        self.items[:self.count] = self.items[-n:] + self.items[:-n]

    def argmin(self) -> int:
        """
        Returns the index of the minimum element in the array.
        """
        if self.count == 0:
            raise ValueError("Array is empty")
        return min(range(self.count), key=lambda i: self.items[i])

    def argmax(self) -> int:
        """
        Returns the index of the maximum element in the array.
        """
        if self.count == 0:
            raise ValueError("Array is empty")
        return max(range(self.count), key=lambda i: self.items[i])

    def reduce_max(self) -> Any:
        """
        Returns the maximum value in the array using a reduce function.
        """
        return self.reduce(lambda a, b: a if a > b else b)

    def reduce_min(self) -> Any:
        """
        Returns the minimum value in the array using a reduce function.
        """
        return self.reduce(lambda a, b: a if a < b else b)

    def reduce_sum(self) -> Any:
        """
        Returns the sum of all values in the array using a reduce function.
        """
        return self.reduce(lambda a, b: a + b)

    def reduce_prod(self) -> Any:
        """
        Returns the product of all values in the array using a reduce function.
        """
        return self.reduce(lambda a, b: a * b)

    def help(self) -> str:
        """
        Returns a help string listing all methods and their descriptions.
        """
        methods = [
            ("__init__", "Initializes the array with a given capacity and fill value."),
            ("__len__", "Returns the number of elements in the array."),
            ("__getitem__", "Returns the item at the given index."),
            ("__setitem__", "Sets the item at the given index to the specified value."),
            ("__contains__", "Returns True if the item is in the array, otherwise False."),
            ("__iter__", "Returns an iterator for the array."),
            ("__repr__", "Returns a string representation of the array."),
            ("__eq__", "Compares this array with another array."),
            ("append", "Appends an item to the end of the array."),
            ("insert", "Inserts an item at the specified index."),
            ("remove", "Removes the first occurrence of the specified item from the array."),
            ("pop", "Removes and returns the item at the specified index (or the last item if no index is specified)."),
            ("clear", "Clears the array."),
            ("index", "Returns the index of the first occurrence of the specified item."),
            ("count", "Returns the number of occurrences of the specified item."),
            ("reverse", "Reverses the array in place."),
            ("extend", "Extends the array by appending elements from the iterable."),
            ("_delete", "Deletes the item at the specified index."),
            ("_resize", "Resizes the internal array to the new capacity."),
            ("sort", "Sorts the array in place."),
            ("binary_search", "Performs binary search on the sorted array and returns the index of the item."),
            ("map", "Applies the given function to each item in the array."),
            ("filter", "Returns a new array containing only the items that satisfy the predicate."),
            ("reduce", "Reduces the array to a single value using the given function."),
            ("min", "Returns the minimum value in the array."),
            ("max", "Returns the maximum value in the array."),
            ("sum", "Returns the sum of all the elements in the array."),
            ("product", "Returns the product of all the elements in the array."),
            ("mean", "Returns the mean (average) of all the elements in the array."),
            ("median", "Returns the median value of the array."),
            ("mode", "Returns the mode (most common value) in the array."),
            ("variance", "Returns the variance of the elements in the array."),
            ("std_dev", "Returns the standard deviation of the elements in the array."),
            ("all", "Returns True if all elements in the array are true, otherwise False."),
            ("any", "Returns True if any element in the array is true, otherwise False."),
            ("is_empty", "Returns True if the array is empty, otherwise False."),
            ("is_sorted", "Returns True if the array is sorted in ascending order, otherwise False."),
            ("copy", "Returns a shallow copy of the array."),
            ("deepcopy", "Returns a deep copy of the array."),
            ("swap", "Swaps the elements at the specified indices."),
            ("rotate_left", "Rotates the array to the left by n steps."),
            ("rotate_right", "Rotates the array to the right by n steps."),
            ("unique", "Returns a new array with unique elements from the original array."),
            ("to_list", "Converts the array to a list."),
            ("from_list", "Initializes the array with elements from the list."),
            ("to_set", "Converts the array to a set."),
            ("to_tuple", "Converts the array to a tuple."),
            ("trim", "Trims the array to its current size."),
            ("compact", "Removes None values from the array."),
            ("fill", "Fills the array with the specified value."),
            ("sample", "Returns a new array with k random elements from the original array."),
            ("flatten", "Flattens the array if it contains nested arrays."),
            ("unflatten", "Converts a flat array into a matrix (2D array) with the given shape."),
            ("chunk", "Splits the array into chunks of the given size."),
            ("apply", "Applies the specified function to each element in the array."),
            ("transform", "Returns a new array where each element is the result of applying the function to the corresponding element."),
            ("percentile", "Returns the pth percentile of the array."),
            ("clip", "Clips the array's values between min_value and max_value."),
            ("roll", "Rolls the array elements by n steps."),
            ("argmin", "Returns the index of the minimum element in the array."),
            ("argmax", "Returns the index of the maximum element in the array."),
            ("reduce_max", "Returns the maximum value in the array using a reduce function."),
            ("reduce_min", "Returns the minimum value in the array using a reduce function."),
            ("reduce_sum", "Returns the sum of all values in the array using a reduce function."),
            ("reduce_prod", "Returns the product of all values in the array using a reduce function."),
            ("help", "Returns a help string listing all methods and their descriptions."),
        ]
        return "\n".join(f"{name}: {desc}" for name, desc in methods)
