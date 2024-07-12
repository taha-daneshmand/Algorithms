from collections import deque
from functools import reduce
import random

class Queue:
    def __init__(self, max_size=None):
        self.items = deque()
        self.max_size = max_size

    def enqueue(self, item):
        if self.max_size is not None and len(self.items) >= self.max_size:
            raise OverflowError("Queue is full")
        self.items.append(item)

    def dequeue(self):
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self.items.popleft()

    def front(self):
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self.items[0]

    def rear(self):
        if self.is_empty():
            raise IndexError("Queue is empty")
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
        return iter(self.items)

    def __repr__(self):
        return f"Queue({', '.join(repr(item) for item in self.items)})"

    def __contains__(self, item):
        return item in self.items

    def copy(self):
        new_queue = Queue(self.max_size)
        new_queue.items = self.items.copy()
        return new_queue

    def to_list(self):
        return list(self.items)

    def from_list(self, lst):
        self.clear()
        for item in lst:
            self.enqueue(item)

    def map(self, func):
        self.items = deque(map(func, self.items))

    def filter(self, pred):
        new_queue = Queue(self.max_size)
        new_queue.items = deque(filter(pred, self.items))
        return new_queue

    def reduce(self, func, initial=None):
        if self.is_empty():
            return initial
        if initial is None:
            return reduce(func, self.items)
        return reduce(func, self.items, initial)

    def rotate(self, n=1):
        self.items.rotate(-n)

    def reverse(self):
        self.items.reverse()

    def extend(self, iterable):
        if self.max_size is not None and len(self.items) + len(iterable) > self.max_size:
            raise OverflowError("Queue size limit exceeded")
        self.items.extend(iterable)

    def count(self, item):
        return self.items.count(item)

    def index(self, item):
        return self.items.index(item)

    def min(self):
        if self.is_empty():
            raise ValueError("Queue is empty")
        return min(self.items)

    def max(self):
        if self.is_empty():
            raise ValueError("Queue is empty")
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
            raise ValueError("Queue is empty")
        return self.sum() / len(self.items)

    def median(self):
        if self.is_empty():
            raise ValueError("Queue is empty")
        sorted_items = sorted(self.items)
        mid = len(sorted_items) // 2
        if len(sorted_items) % 2 == 0:
            return (sorted_items[mid - 1] + sorted_items[mid]) / 2
        return sorted_items[mid]

    def mode(self):
        if self.is_empty():
            raise ValueError("Queue is empty")
        frequency = {}
        for item in self.items:
            frequency[item] = frequency.get(item, 0) + 1
        max_count = max(frequency.values())
        modes = [k for k, v in frequency.items() if v == max_count]
        return modes if len(modes) > 1 else modes[0]

    def variance(self):
        if self.is_empty():
            raise ValueError("Queue is empty")
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
        new_queue = Queue(self.max_size)
        for item in self.items:
            if item not in seen:
                seen.add(item)
                new_queue.enqueue(item)
        return new_queue

    def apply(self, func):
        new_queue = Queue(self.max_size)
        for item in self.items:
            new_queue.enqueue(func(item))
        return new_queue

    def transform(self, func):
        new_queue = Queue(self.max_size)
        for item in self.items:
            new_queue.enqueue(func(item))
        return new_queue

    def shuffle(self):
        temp_list = list(self.items)
        random.shuffle(temp_list)
        self.items = deque(temp_list)

    def sample(self, k):
        if k > len(self.items):
            raise ValueError("Sample size larger than population")
        return random.sample(self.items, k)

    def sort(self, key=None, reverse=False):
        self.items = deque(sorted(self.items, key=key, reverse=reverse))

    def split(self, n):
        if n <= 0 or n > len(self.items):
            raise ValueError("Invalid split size")
        part1 = deque()
        part2 = deque()
        for i in range(n):
            part1.append(self.items.popleft())
        while self.items:
            part2.append(self.items.popleft())
        return Queue.from_list(part1), Queue.from_list(part2)

    def split_half(self):
        return self.split(len(self.items) // 2)

    def merge(self, other):
        if not isinstance(other, Queue):
            raise ValueError("Can only merge with another Queue")
        if self.max_size is not None and len(self.items) + len(other.items) > self.max_size:
            raise OverflowError("Queue size limit exceeded")
        self.items.extend(other.items)

    def intersect(self, other):
        if not isinstance(other, Queue):
            raise ValueError("Can only intersect with another Queue")
        new_queue = Queue(self.max_size)
        for item in self.items:
            if item in other.items:
                new_queue.enqueue(item)
        return new_queue

    def difference(self, other):
        if not isinstance(other, Queue):
            raise ValueError("Can only difference with another Queue")
        new_queue = Queue(self.max_size)
        for item in self.items:
            if item not in other.items:
                new_queue.enqueue(item)
        return new_queue

    def symmetric_difference(self, other):
        if not isinstance(other, Queue):
            raise ValueError("Can only symmetric difference with another Queue")
        return self.difference(other).merge(other.difference(self))

    def to_set(self):
        return set(self.items)

    def extendleft(self, iterable):
        if self.max_size is not None and len(self.items) + len(iterable) > self.max_size:
            raise OverflowError("Queue size limit exceeded")
        self.items.extendleft(iterable)

    def popleft(self):
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self.items.popleft()

    def popright(self):
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self.items.pop()

    def rotate_left(self, n=1):
        self.items.rotate(-n)

    def rotate_right(self, n=1):
        self.items.rotate(n)

    def __add__(self, other):
        if not isinstance(other, Queue):
            raise ValueError("Can only add another Queue")
        new_queue = self.copy()
        new_queue.merge(other)
        return new_queue

    def __iadd__(self, other):
        self.merge(other)
        return self

    def __eq__(self, other):
        if not isinstance(other, Queue):
            return False
        return list(self.items) == list(other.items)

    def __ne__(self, other):
        return not self == other

    def __gt__(self, other):
        if not isinstance(other, Queue):
            return False
        return list(self.items) > list(other.items)

    def __lt__(self, other):
        if not isinstance(other, Queue):
            return False
        return list(self.items) < list(other.items)

    def __ge__(self, other):
        return not self < other

    def __le__(self, other):
        return not self > other
