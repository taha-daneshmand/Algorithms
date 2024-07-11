from collections import deque

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
