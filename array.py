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
