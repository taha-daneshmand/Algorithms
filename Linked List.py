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
