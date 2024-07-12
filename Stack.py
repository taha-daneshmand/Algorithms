class Stack:
    def __init__(self, max_size=None):
        self.items = []
        self.max_size = max_size

    def push(self, item):
        if self.max_size is not None and len(self.items) >= self.max_size:
            raise OverflowError("Stack is full")
        self.items.append(item)

    def pop(self):
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.items.pop()

    def peek(self):
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.items[-1]

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)

    def clear(self):
        self.items.clear()

    def __len__(self):
        return self.size()

    def __iter__(self):
        return reversed(self.items)

    def __repr__(self):
        return f"Stack({', '.join(repr(item) for item in self)})"

    def __contains__(self, item):
        return item in self.items

    def copy(self):
        new_stack = Stack(self.max_size)
        new_stack.items = self.items.copy()
        return new_stack

    def reverse(self):
        self.items.reverse()

    def to_list(self):
        return list(self)

    def from_list(self, lst):
        self.clear()
        for item in reversed(lst):
            self.push(item)

    def map(self, func):
        self.items = [func(item) for item in reversed(self.items)]
        self.items.reverse()

    def filter(self, pred):
        new_stack = Stack(self.max_size)
        for item in self:
            if pred(item):
                new_stack.push(item)
        return new_stack

    def reduce(self, func, initial=None):
        if self.is_empty():
            return initial
        if initial is None:
            result = self.items[-1]
            items = self.items[:-1]
        else:
            result = initial
            items = self.items
        for item in reversed(items):
            result = func(result, item)
        return result

    def min(self):
        if self.is_empty():
            raise ValueError("Stack is empty")
        return min(self.items)

    def max(self):
        if self.is_empty():
            raise ValueError("Stack is empty")
        return max(self.items)

    def sum(self):
        return sum(self.items)

    def product(self):
        result = 1
        for item in self.items:
            result *= item
        return result

    def mean(self):
        if self.is_empty():
            raise ValueError("Stack is empty")
        return self.sum() / len(self.items)

    def median(self):
        if self.is_empty():
            raise ValueError("Stack is empty")
        sorted_items = sorted(self.items)
        mid = len(sorted_items) // 2
        if len(sorted_items) % 2 == 0:
            return (sorted_items[mid - 1] + sorted_items[mid]) / 2
        return sorted_items[mid]

    def mode(self):
        if self.is_empty():
            raise ValueError("Stack is empty")
        frequency = {}
        for item in self.items:
            frequency[item] = frequency.get(item, 0) + 1
        max_count = max(frequency.values())
        modes = [k for k, v in frequency.items() if v == max_count]
        return modes if len(modes) > 1 else modes[0]

    def variance(self):
        if self.is_empty():
            raise ValueError("Stack is empty")
        mean_value = self.mean()
        return sum((x - mean_value) ** 2 for x in self.items) / len(self.items)

    def std_dev(self):
        return self.variance() ** 0.5

    def all(self):
        return all(self.items)

    def any(self):
        return any(self.items)

    def unique(self):
        seen = set()
        new_stack = Stack(self.max_size)
        for item in self.items:
            if item not in seen:
                seen.add(item)
                new_stack.push(item)
        return new_stack

    def apply(self, func):
        new_stack = Stack(self.max_size)
        for item in self.items:
            new_stack.push(func(item))
        return new_stack

    def shuffle(self):
        temp_list = self.items.copy()
        random.shuffle(temp_list)
        self.items = temp_list

    def sample(self, k):
        if k > len(self.items):
            raise ValueError("Sample size larger than population")
        return random.sample(self.items, k)

    def sort(self, key=None, reverse=False):
        self.items.sort(key=key, reverse=reverse)

    def extend(self, iterable):
        if self.max_size is not None and len(self.items) + len(iterable) > self.max_size:
            raise OverflowError("Stack size limit exceeded")
        for item in iterable:
            self.push(item)

    def count(self, item):
        return self.items.count(item)

    def index(self, item):
        return self.items.index(item)

    def rotate_left(self, n=1):
        self.items = self.items[n:] + self.items[:n]

    def rotate_right(self, n=1):
        self.items = self.items[-n:] + self.items[:-n]

    def __add__(self, other):
        if not isinstance(other, Stack):
            raise ValueError("Can only add another Stack")
        new_stack = self.copy()
        for item in other.items:
            new_stack.push(item)
        return new_stack

    def __iadd__(self, other):
        if not isinstance(other, Stack):
            raise ValueError("Can only add another Stack")
        for item in other.items:
            self.push(item)
        return self

    def __eq__(self, other):
        if not isinstance(other, Stack):
            return False
        return self.items == other.items

    def __ne__(self, other):
        return not self == other

    def __gt__(self, other):
        if not isinstance(other, Stack):
            return False
        return self.items > other.items

    def __lt__(self, other):
        if not isinstance(other, Stack):
            return False
        return self.items < other.items

    def __ge__(self, other):
        return not self < other

    def __le__(self, other):
        return not self > other

    def flatten(self):
        flat_stack = Stack(self.max_size)
        def _flatten(lst):
            for item in lst:
                if isinstance(item, list):
                    _flatten(item)
                else:
                    flat_stack.push(item)
        _flatten(self.items)
        return flat_stack

    def swap(self, i, j):
        if i >= len(self.items) or j >= len(self.items):
            raise IndexError("Index out of range")
        self.items[i], self.items[j] = self.items[j], self.items[i]

    def insert(self, index, item):
        if self.max_size is not None and len(self.items) >= self.max_size:
            raise OverflowError("Stack is full")
        self.items.insert(index, item)

    def remove(self, item):
        self.items.remove(item)

    def __mul__(self, n):
        if not isinstance(n, int):
            raise ValueError("Can only multiply by an integer")
        new_stack = Stack(self.max_size)
        new_stack.items = self.items * n
        return new_stack

    def __imul__(self, n):
        if not isinstance(n, int):
            raise ValueError("Can only multiply by an integer")
        self.items *= n
        return self

    def __pow__(self, n):
        if not isinstance(n, int):
            raise ValueError("Can only exponentiate by an integer")
        new_stack = Stack(self.max_size)
        new_stack.items = self.items ** n
        return new_stack

    def __ipow__(self, n):
        if not isinstance(n, int):
            raise ValueError("Can only exponentiate by an integer")
        self.items **= n
        return self
