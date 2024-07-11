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
