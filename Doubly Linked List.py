class Node:
    def __init__(self, data):
        self.data = data
        self.prev = None
        self.next = None

class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0

    def append(self, data):
        """Adds a node to the end of the list."""
        new_node = Node(data)
        if not self.head:
            self.head = self.tail = new_node
        else:
            self.tail.next = new_node
            new_node.prev = self.tail
            self.tail = new_node
        self.size += 1

    def prepend(self, data):
        """Adds a node to the beginning of the list."""
        new_node = Node(data)
        if not self.head:
            self.head = self.tail = new_node
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
        self.size += 1

    def insert(self, index, data):
        """Inserts a node at the given index."""
        if index < 0 or index > self.size:
            raise IndexError("Index out of bounds")
        if index == 0:
            self.prepend(data)
        elif index == self.size:
            self.append(data)
        else:
            new_node = Node(data)
            current = self.head
            for _ in range(index - 1):
                current = current.next
            new_node.next = current.next
            new_node.prev = current
            current.next.prev = new_node
            current.next = new_node
            self.size += 1

    def remove(self, data):
        """Removes the first node with the specified data."""
        current = self.head
        while current:
            if current.data == data:
                if current.prev:
                    current.prev.next = current.next
                else:
                    self.head = current.next
                if current.next:
                    current.next.prev = current.prev
                else:
                    self.tail = current.prev
                self.size -= 1
                return
            current = current.next
        raise ValueError("Data not found in the list")

    def pop(self, index=None):
        """Removes and returns the node at the specified index."""
        if self.is_empty():
            raise IndexError("Pop from empty list")
        if index is None:
            index = self.size - 1
        if index < 0 or index >= self.size:
            raise IndexError("Index out of bounds")
        current = self.head
        for _ in range(index):
            current = current.next
        if current.prev:
            current.prev.next = current.next
        else:
            self.head = current.next
        if current.next:
            current.next.prev = current.prev
        else:
            self.tail = current.prev
        self.size -= 1
        return current.data

    def reverse(self):
        """Reverses the list in place."""
        current = self.head
        self.head, self.tail = self.tail, self.head
        while current:
            current.prev, current.next = current.next, current.prev
            current = current.prev

    def is_empty(self):
        """Checks if the list is empty."""
        return self.size == 0

    def clear(self):
        """Clears the list."""
        self.head = None
        self.tail = None
        self.size = 0

    def find(self, data):
        """Finds and returns the first node containing the specified data."""
        current = self.head
        while current:
            if current.data == data:
                return current
            current = current.next
        return None

    def index_of(self, data):
        """Returns the index of the first occurrence of the specified data."""
        current = self.head
        index = 0
        while current:
            if current.data == data:
                return index
            current = current.next
            index += 1
        raise ValueError("Data not found in the list")

    def contains(self, data):
        """Checks if the list contains the specified data."""
        return self.find(data) is not None

    def to_list(self):
        """Converts the linked list to a Python list."""
        result = []
        current = self.head
        while current:
            result.append(current.data)
            current = current.next
        return result

    def from_list(self, lst):
        """Populates the linked list from a Python list."""
        self.clear()
        for item in lst:
            self.append(item)

    def sort(self):
        """Sorts the linked list in ascending order."""
        if self.size > 1:
            current = self.head
            while current:
                index = current.next
                while index:
                    if current.data > index.data:
                        current.data, index.data = index.data, current.data
                    index = index.next
                current = current.next

    def unique(self):
        """Removes duplicate elements from the list."""
        seen = set()
        current = self.head
        while current:
            if current.data in seen:
                next_node = current.next
                self.remove(current.data)
                current = next_node
            else:
                seen.add(current.data)
                current = current.next

    def rotate_left(self, n):
        """Rotates the list to the left by n positions."""
        if self.size <= 1 or n <= 0:
            return
        n = n % self.size
        for _ in range(n):
            self.append(self.pop(0))

    def rotate_right(self, n):
        """Rotates the list to the right by n positions."""
        if self.size <= 1 or n <= 0:
            return
        n = n % self.size
        for _ in range(n):
            self.prepend(self.pop())

    def split(self, index):
        """Splits the list into two at the specified index."""
        if index < 0 or index > self.size:
            raise IndexError("Index out of bounds")
        list1 = DoublyLinkedList()
        list2 = DoublyLinkedList()
        current = self.head
        for i in range(self.size):
            if i < index:
                list1.append(current.data)
            else:
                list2.append(current.data)
            current = current.next
        return list1, list2

    def merge(self, other):
        """Merges another doubly linked list into this list."""
        if not isinstance(other, DoublyLinkedList):
            raise ValueError("Can only merge with another DoublyLinkedList")
        current = other.head
        while current:
            self.append(current.data)
            current = current.next

    def intersect(self, other):
        """Finds the intersection of two doubly linked lists."""
        if not isinstance(other, DoublyLinkedList):
            raise ValueError("Can only intersect with another DoublyLinkedList")
        result = DoublyLinkedList()
        current = self.head
        while current:
            if other.contains(current.data):
                result.append(current.data)
            current = current.next
        return result

    def map(self, func):
        """Applies a function to each element in the list."""
        current = self.head
        while current:
            current.data = func(current.data)
            current = current.next

    def filter(self, pred):
        """Filters the list based on a predicate."""
        current = self.head
        while current:
            next_node = current.next
            if not pred(current.data):
                self.remove(current.data)
            current = next_node

    def sum(self):
        """Calculates the sum of all elements in the list."""
        total = 0
        current = self.head
        while current:
            total += current.data
            current = current.next
        return total

    def product(self):
        """Calculates the product of all elements in the list."""
        result = 1
        current = self.head
        while current:
            result *= current.data
            current = current.next
        return result

    def mean(self):
        """Calculates the mean of all elements in the list."""
        if self.is_empty():
            raise ValueError("Cannot calculate mean of an empty list")
        return self.sum() / self.size

    def max(self):
        """Finds the maximum value in the list."""
        if self.is_empty():
            raise ValueError("List is empty")
        max_value = self.head.data
        current = self.head
        while current:
            if current.data > max_value:
                max_value = current.data
            current = current.next
        return max_value

    def min(self):
        """Finds the minimum value in the list."""
        if self.is_empty():
            raise ValueError("List is empty")
        min_value = self.head.data
        current = self.head
        while current:
            if current.data < min_value:
                min_value = current.data
            current = current.next
        return min_value

    def kth_from_end(self, k):
        """Finds the kth element from the end of the list."""
        if k <= 0 or k > self.size:
            raise IndexError("Index out of bounds")
        current = self.tail
        for _ in range(k - 1):
            current = current.prev
        return current.data

    def __len__(self):
        return self.size

    def __iter__(self):
        current = self.head
        while current:
            yield current.data
            current = current.next

    def __repr__(self):
        return f"DoublyLinkedList([{', '.join(map(str, self))}])"
