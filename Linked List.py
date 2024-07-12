class Node:
    def __init__(self, data):
        self.data = data
        self.prev = None
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0

    def __len__(self):
        return self.size

    def is_empty(self):
        return self.size == 0

    def append(self, data):
        new_node = Node(data)
        if self.is_empty():
            self.head = self.tail = new_node
        else:
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node
        self.size += 1

    def prepend(self, data):
        new_node = Node(data)
        if self.is_empty():
            self.head = self.tail = new_node
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
        self.size += 1

    def insert(self, index, data):
        if index < 0 or index > self.size:
            raise IndexError("Invalid index")
        if index == 0:
            self.prepend(data)
        elif index == self.size:
            self.append(data)
        else:
            new_node = Node(data)
            current = self._get_node(index)
            new_node.prev = current.prev
            new_node.next = current
            current.prev.next = new_node
            current.prev = new_node
            self.size += 1

    def remove(self, data):
        current = self.head
        while current:
            if current.data == data:
                self._remove_node(current)
                return True
            current = current.next
        return False

    def pop(self, index=None):
        if self.is_empty():
            raise IndexError("List is empty")
        if index is None:
            index = self.size - 1
        if index < 0 or index >= self.size:
            raise IndexError("Invalid index")
        node = self._get_node(index)
        self._remove_node(node)
        return node.data

    def _remove_node(self, node):
        if node.prev:
            node.prev.next = node.next
        else:
            self.head = node.next
        if node.next:
            node.next.prev = node.prev
        else:
            self.tail = node.prev
        self.size -= 1

    def _get_node(self, index):
        if index < self.size // 2:
            current = self.head
            for _ in range(index):
                current = current.next
        else:
            current = self.tail
            for _ in range(self.size - 1, index, -1):
                current = current.prev
        return current

    def __getitem__(self, index):
        return self._get_node(index).data

    def __setitem__(self, index, data):
        self._get_node(index).data = data

    def index(self, data):
        current = self.head
        for i in range(self.size):
            if current.data == data:
                return i
            current = current.next
        raise ValueError("Data not found in the list")

    def count(self, data):
        count = 0
        current = self.head
        while current:
            if current.data == data:
                count += 1
            current = current.next
        return count

    def reverse(self):
        current = self.head
        self.head, self.tail = self.tail, self.head
        while current:
            current.prev, current.next = current.next, current.prev
            current = current.prev

    def __iter__(self):
        current = self.head
        while current:
            yield current.data
            current = current.next

    def __repr__(self):
        return f"LinkedList({', '.join(str(item) for item in self)})"

    def clear(self):
        self.head = None
        self.tail = None
        self.size = 0

    def copy(self):
        new_list = LinkedList()
        for item in self:
            new_list.append(item)
        return new_list

    def extend(self, iterable):
        for item in iterable:
            self.append(item)

    def sort(self, key=None, reverse=False):
        sorted_data = sorted(self, key=key, reverse=reverse)
        self.clear()
        for item in sorted_data:
            self.append(item)

    def map(self, func):
        current = self.head
        while current:
            current.data = func(current.data)
            current = current.next

    def filter(self, pred):
        new_list = LinkedList()
        for item in self:
            if pred(item):
                new_list.append(item)
        return new_list

    def reduce(self, func, initial=None):
        if self.is_empty():
            return initial
        if initial is None:
            result = self.head.data
            current = self.head.next
        else:
            result = initial
            current = self.head
        while current:
            result = func(result, current.data)
            current = current.next
        return result

    def min(self):
        if self.is_empty():
            raise ValueError("List is empty")
        min_value = self.head.data
        current = self.head.next
        while current:
            if current.data < min_value:
                min_value = current.data
            current = current.next
        return min_value

    def max(self):
        if self.is_empty():
            raise ValueError("List is empty")
        max_value = self.head.data
        current = self.head.next
        while current:
            if current.data > max_value:
                max_value = current.data
            current = current.next
        return max_value

    def sum(self):
        total = 0
        current = self.head
        while current:
            total += current.data
            current = current.next
        return total

    def product(self):
        result = 1
        current = self.head
        while current:
            result *= current.data
            current = current.next
        return result

    def mean(self):
        if self.is_empty():
            raise ValueError("List is empty")
        return self.sum() / self.size

    def median(self):
        if self.is_empty():
            raise ValueError("List is empty")
        sorted_data = sorted(self)
        mid = self.size // 2
        if self.size % 2 == 0:
            return (sorted_data[mid - 1] + sorted_data[mid]) / 2
        return sorted_data[mid]

    def mode(self):
        if self.is_empty():
            raise ValueError("List is empty")
        frequency = {}
        current = self.head
        while current:
            frequency[current.data] = frequency.get(current.data, 0) + 1
            current = current.next
        max_count = max(frequency.values())
        modes = [k for k, v in frequency.items() if v == max_count]
        return modes if len(modes) > 1 else modes[0]

    def variance(self):
        if self.is_empty():
            raise ValueError("List is empty")
        mean_value = self.mean()
        return sum((x - mean_value) ** 2 for x in self) / self.size

    def std_dev(self):
        return self.variance() ** 0.5

    def all(self):
        return all(self)

    def any(self):
        return any(self)

    def is_sorted(self):
        current = self.head
        while current and current.next:
            if current.data > current.next.data:
                return False
            current = current.next
        return True

    def swap(self, i, j):
        if i < 0 or i >= self.size or j < 0 or j >= self.size:
            raise IndexError("Invalid index")
        node_i = self._get_node(i)
        node_j = self._get_node(j)
        node_i.data, node_j.data = node_j.data, node_i.data

    def rotate_left(self, n):
        if self.is_empty() or n == 0:
            return
        n = n % self.size
        for _ in range(n):
            data = self.pop(0)
            self.append(data)

    def rotate_right(self, n):
        if self.is_empty() or n == 0:
            return
        n = n % self.size
        for _ in range(n):
            data = self.pop(self.size - 1)
            self.prepend(data)

    def unique(self):
        seen = set()
        current = self.head
        while current:
            if current.data in seen:
                next_node = current.next
                self._remove_node(current)
                current = next_node
            else:
                seen.add(current.data)
                current = current.next

    def to_list(self):
        return list(self)

    def from_list(self, lst):
        self.clear()
        for item in lst:
            self.append(item)

    def sample(self, k):
        import random
        if k > self.size:
            raise ValueError("Sample larger than population")
        indices = random.sample(range(self.size), k)
        return [self[i] for i in indices]

    def apply(self, func):
        current = self.head
        while current:
            current.data = func(current.data)
            current = current.next

    def transform(self, func):
        new_list = LinkedList()
        current = self.head
        while current:
            new_list.append(func(current.data))
            current = current.next
        return new_list

    def reduce_max(self):
        return self.reduce(lambda a, b: a if a > b else b)

    def reduce_min(self):
        return self.reduce(lambda a, b: a if a < b else b)

    def reduce_sum(self):
        return self.reduce(lambda a, b: a + b)

    def reduce_prod(self):
        return self.reduce(lambda a, b: a * b)
