import math
import random
import time

class HashTable:
    def __init__(self, size=100, hash_func=None):
        self.size = size
        self.table = [[] for _ in range(self.size)]
        self.hash_func = hash_func or (lambda key: hash(key) % self.size)
        self.item_count = 0

    def __setitem__(self, key, value):
        index = self.hash_func(key)
        for item in self.table[index]:
            if item[0] == key:
                item[1] = value
                return
        self.table[index].append([key, value])
        self.item_count += 1
        if self.item_count > self.size * 0.7:
            self._resize()

    def __getitem__(self, key):
        index = self.hash_func(key)
        for item in self.table[index]:
            if item[0] == key:
                return item[1]
        raise KeyError(key)

    def __delitem__(self, key):
        index = self.hash_func(key)
        for i, item in enumerate(self.table[index]):
            if item[0] == key:
                del self.table[index][i]
                self.item_count -= 1
                return
        raise KeyError(key)

    def __contains__(self, key):
        try:
            self[key]
            return True
        except KeyError:
            return False

    def __len__(self):
        return self.item_count

    def __iter__(self):
        for bucket in self.table:
            for key, _ in bucket:
                yield key

    def __str__(self):
        return str(dict(self.items()))

    def __repr__(self):
        return f"{self.__class__.__name__}({dict(self.items())})"

    def __eq__(self, other):
        if not isinstance(other, HashTable):
            return False
        return len(self) == len(other) and all(self[k] == other[k] for k in self)

    def __ne__(self, other):
        return not self.__eq__(other)

    def _resize(self):
        new_size = self.size * 2
        new_table = HashTable(new_size, self.hash_func)
        for key in self:
            new_table[key] = self[key]
        self.size = new_size
        self.table = new_table.table

    def keys(self):
        return list(self)

    def values(self):
        return [self[key] for key in self]

    def items(self):
        return [(key, self[key]) for key in self]

    def clear(self):
        self.table = [[] for _ in range(self.size)]
        self.item_count = 0

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def update(self, other=None, **kwargs):
        if other:
            if isinstance(other, dict):
                for key, value in other.items():
                    self[key] = value
            else:
                for key, value in other:
                    self[key] = value
        for key, value in kwargs.items():
            self[key] = value

    def pop(self, key, default=object()):
        try:
            value = self[key]
            del self[key]
            return value
        except KeyError:
            if default is object():
                raise KeyError(key)
            return default

    def setdefault(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            self[key] = default
            return default

    def copy(self):
        new_hash_table = HashTable(self.size, self.hash_func)
        for key, value in self.items():
            new_hash_table[key] = value
        return new_hash_table

    @classmethod
    def fromkeys(cls, iterable, value=None):
        hash_table = cls()
        for key in iterable:
            hash_table[key] = value
        return hash_table

    def popitem(self):
        if not self:
            raise KeyError("HashTable is empty")
        key = next(iter(self))
        value = self[key]
        del self[key]
        return (key, value)

    def to_dict(self):
        return dict(self.items())

    def from_dict(self, dictionary):
        self.clear()
        self.update(dictionary)

    def merge(self, other):
        for key, value in other.items():
            if key not in self:
                self[key] = value
        return self

    def filter(self, predicate):
        return HashTable({k: v for k, v in self.items() if predicate(k, v)})

    def map_values(self, func):
        return HashTable({k: func(v) for k, v in self.items()})

    def map_keys(self, func):
        return HashTable({func(k): v for k, v in self.items()})

    def invert(self):
        return HashTable({v: k for k, v in self.items()})

    def keys_starting_with(self, prefix):
        return [k for k in self.keys() if str(k).startswith(prefix)]

    def values_above(self, threshold):
        return [v for v in self.values() if v > threshold]

    def keys_containing(self, substring):
        return [k for k in self.keys() if substring in str(k)]

    def most_common_value(self):
        if not self:
            return None
        return max(set(self.values()), key=self.values().count)

    def least_common_value(self):
        if not self:
            return None
        return min(set(self.values()), key=self.values().count)

    def key_with_max_value(self):
        if not self:
            return None
        return max(self, key=self.get)

    def key_with_min_value(self):
        if not self:
            return None
        return min(self, key=self.get)

    def sum_values(self):
        return sum(self.values())

    def product_values(self):
        result = 1
        for value in self.values():
            result *= value
        return result

    def average_value(self):
        if not self:
            return 0
        return self.sum_values() / len(self)

    def median_value(self):
        values = sorted(self.values())
        length = len(values)
        if length % 2 == 0:
            return (values[length//2 - 1] + values[length//2]) / 2
        return values[length//2]

    def mode_value(self):
        if not self:
            return None
        return max(set(self.values()), key=self.values().count)

    def variance_value(self):
        if len(self) < 2:
            return 0
        avg = self.average_value()
        return sum((x - avg) ** 2 for x in self.values()) / (len(self) - 1)

    def std_dev_value(self):
        return math.sqrt(self.variance_value())

    def range_value(self):
        if not self:
            return 0
        values = self.values()
        return max(values) - min(values)

    def percentile_value(self, percentile):
        values = sorted(self.values())
        index = (len(values) - 1) * percentile / 100
        if index.is_integer():
            return values[int(index)]
        else:
            return (values[int(index)] + values[int(index) + 1]) / 2

    def keys_of_type(self, key_type):
        return [k for k in self.keys() if isinstance(k, key_type)]

    def values_of_type(self, value_type):
        return [v for v in self.values() if isinstance(v, value_type)]

    def items_of_type(self, key_type, value_type):
        return [(k, v) for k, v in self.items() if isinstance(k, key_type) and isinstance(v, value_type)]

    def keys_by_value(self, value):
        return [k for k, v in self.items() if v == value]

    def remove_by_value(self, value):
        keys_to_remove = self.keys_by_value(value)
        for key in keys_to_remove:
            del self[key]

    def keep_only(self, keys):
        keys_to_remove = set(self.keys()) - set(keys)
        for key in keys_to_remove:
            del self[key]

    def remove_keys(self, keys):
        for key in keys:
            self.pop(key, None)

    def rename_key(self, old_key, new_key):
        if old_key in self:
            self[new_key] = self.pop(old_key)

    def swap_keys(self, key1, key2):
        if key1 in self and key2 in self:
            self[key1], self[key2] = self[key2], self[key1]

    def is_subset(self, other):
        return all(key in other and self[key] == other[key] for key in self)

    def is_superset(self, other):
        return all(key in self and self[key] == other[key] for key in other)

    def is_disjoint(self, other):
        return not any(key in self for key in other)

    def union(self, other):
        new_hash_table = self.copy()
        new_hash_table.update(other)
        return new_hash_table

    def intersection(self, other):
        return HashTable({k: self[k] for k in self if k in other and self[k] == other[k]})

    def difference(self, other):
        return HashTable({k: self[k] for k in self if k not in other})

    def symmetric_difference(self, other):
        new_hash_table = self.difference(other)
        new_hash_table.update(other.difference(self))
        return new_hash_table

    def increment(self, key, amount=1):
        self[key] = self.get(key, 0) + amount

    def decrement(self, key, amount=1):
        self[key] = self.get(key, 0) - amount

    def multiply(self, key, factor):
        if key in self:
            self[key] *= factor

    def divide(self, key, divisor):
        if key in self and divisor != 0:
            self[key] /= divisor

    def apply_to_values(self, func):
        for key in self:
            self[key] = func(self[key])

    def apply_to_keys(self, func):
        new_items = [(func(k), v) for k, v in self.items()]
        self.clear()
        for k, v in new_items:
            self[k] = v

    def group_by_value(self):
        result = {}
        for key, value in self.items():
            if value not in result:
                result[value] = []
            result[value].append(key)
        return HashTable(result)

    def flip_dict(self):
        return HashTable({v: k for k, v in self.items()})

    def prefix_keys(self, prefix):
        return HashTable({f"{prefix}{k}": v for k, v in self.items()})

    def suffix_keys(self, suffix):
        return HashTable({f"{k}{suffix}": v for k, v in self.items()})

    def remove_prefix(self, prefix):
        return HashTable({k[len(prefix):]: v for k, v in self.items() if k.startswith(prefix)})

    def remove_suffix(self, suffix):
        return HashTable({k[:-len(suffix)]: v for k, v in self.items() if k.endswith(suffix)})

    def keys_to_lowercase(self):
        return HashTable({str(k).lower(): v for k, v in self.items()})

    def keys_to_uppercase(self):
        return HashTable({str(k).upper(): v for k, v in self.items()})

    def capitalize_keys(self):
        return HashTable({str(k).capitalize(): v for k, v in self.items()})

    def sort_by_keys(self):
        return HashTable(sorted(self.items()))

    def sort_by_values(self):
        return HashTable(sorted(self.items(), key=lambda x: x[1]))

    def reverse(self):
        return HashTable(reversed(self.items()))

    def first_n_items(self, n):
        return HashTable(list(self.items())[:n])

    def last_n_items(self, n):
        return HashTable(list(self.items())[-n:])

    def sample(self, n):
        return HashTable(random.sample(list(self.items()), min(n, len(self))))

    def shuffle(self):
        items = list(self.items())
        random.shuffle(items)
        return HashTable(items)

    def chunk(self, size):
        items = list(self.items())
        return [HashTable(items[i:i+size]) for i in range(0, len(items), size)]

    def split(self, n):
        items = list(self.items())
        chunk_size = len(items) // n
        return [HashTable(items[i:i+chunk_size]) for i in range(0, len(items), chunk_size)]

    def rotate(self, n):
        items = list(self.items())
        n = n % len(items)
        return HashTable(items[n:] + items[:n])

    def find_key(self, value):
        for k, v in self.items():
            if v == value:
                return k
        raise ValueError("Value not found")

    def find_all_keys(self, value):
        return [k for k, v in self.items() if v == value]

    def count_value(self, value):
        return sum(1 for v in self.values() if v == value)

    def count_key(self, key):
        return 1 if key in self else 0

    def unique_values(self):
        return list(set(self.values()))

    def unique_keys(self):
        return list(set(self.keys()))

    def key_frequencies(self):
        return HashTable({k: 1 for k in self})

    def value_frequencies(self):
        freq = {}
        for v in self.values():
            freq[v] = freq.get(v, 0) + 1
        return HashTable(freq)

    def flatten(self):
        flattened = []
        for k, v in self.items():
            flattened.extend([k, v])
        return flattened

    def unflatten(self, flat_list):
        return HashTable({flat_list[i]: flat_list[i+1] for i in range(0, len(flat_list), 2)})

    def deep_copy(self):
        def deep_copy_value(value):
            if isinstance(value, (list, tuple)):
                return type(value)(deep_copy_value(v) for v in value)
            elif isinstance(value, dict):
                return {k: deep_copy_value(v) for k, v in value.items()}
            elif isinstance(value, set):
                return {deep_copy_value(v)}
