class Node:
    def __init__(self, key):
        self.left = None
        self.right = None
        self.val = key

class BinaryTree:
    def __init__(self):
        self.root = None

    def insert(self, key):
        if self.root is None:
            self.root = Node(key)
        else:
            self._insert(self.root, key)

    def _insert(self, root, key):
        if key < root.val:
            if root.left is None:
                root.left = Node(key)
            else:
                self._insert(root.left, key)
        else:
            if root.right is None:
                root.right = Node(key)
            else:
                self._insert(root.right, key)

    def search(self, key):
        return self._search(self.root, key)

    def _search(self, root, key):
        if root is None or root.val == key:
            return root
        if key < root.val:
            return self._search(root.left, key)
        return self._search(root.right, key)

    def delete(self, key):
        self.root = self._delete(self.root, key)

    def _delete(self, root, key):
        if root is None:
            return root
        if key < root.val:
            root.left = self._delete(root.left, key)
        elif key > root.val:
            root.right = self._delete(root.right, key)
        else:
            if root.left is None:
                return root.right
            elif root.right is None:
                return root.left
            temp = self._min_value_node(root.right)
            root.val = temp.val
            root.right = self._delete(root.right, temp.val)
        return root

    def _min_value_node(self, node):
        current = node
        while current.left is not None:
            current = current.left
        return current

    def inorder(self):
        return self._inorder(self.root, [])

    def _inorder(self, root, res):
        if root:
            self._inorder(root.left, res)
            res.append(root.val)
            self._inorder(root.right, res)
        return res

    def preorder(self):
        return self._preorder(self.root, [])

    def _preorder(self, root, res):
        if root:
            res.append(root.val)
            self._preorder(root.left, res)
            self._preorder(root.right, res)
        return res

    def postorder(self):
        return self._postorder(self.root, [])

    def _postorder(self, root, res):
        if root:
            self._postorder(root.left, res)
            self._postorder(root.right, res)
            res.append(root.val)
        return res

    def level_order(self):
        if self.root is None:
            return []
        res, queue = [], [self.root]
        while queue:
            node = queue.pop(0)
            res.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        return res

    def height(self):
        return self._height(self.root)

    def _height(self, root):
        if root is None:
            return 0
        return max(self._height(root.left), self._height(root.right)) + 1

    def count_nodes(self):
        return self._count_nodes(self.root)

    def _count_nodes(self, root):
        if root is None:
            return 0
        return 1 + self._count_nodes(root.left) + self._count_nodes(root.right)

    def is_balanced(self):
        def check(root):
            if root is None:
                return 0, True
            left_height, left_balanced = check(root.left)
            right_height, right_balanced = check(root.right)
            balanced = left_balanced and right_balanced and abs(left_height - right_height) <= 1
            return max(left_height, right_height) + 1, balanced
        _, result = check(self.root)
        return result

    def is_symmetric(self):
        def is_mirror(left, right):
            if left is None and right is None:
                return True
            if left is None or right is None:
                return False
            return (left.val == right.val) and is_mirror(left.left, right.right) and is_mirror(left.right, right.left)
        return is_mirror(self.root, self.root)

    def paths(self):
        def find_paths(root, path, paths):
            if root is None:
                return
            path.append(root.val)
            if root.left is None and root.right is None:
                paths.append(list(path))
            else:
                find_paths(root.left, path, paths)
                find_paths(root.right, path, paths)
            path.pop()
        paths = []
        find_paths(self.root, [], paths)
        return paths

    def find_path(self, key):
        def path_to_node(root, path, key):
            if root is None:
                return False
            path.append(root.val)
            if root.val == key:
                return True
            if (root.left and path_to_node(root.left, path, key)) or (root.right and path_to_node(root.right, path, key)):
                return True
            path.pop()
            return False
        path = []
        if path_to_node(self.root, path, key):
            return path
        return None

    def __contains__(self, key):
        return self.search(key) is not None

    def __iter__(self):
        return iter(self.inorder())

    def __repr__(self):
        return f"BinaryTree({self.inorder()})"

    # Additional methods to extend functionality
    def is_full(self):
        def check_full(root):
            if root is None:
                return True
            if root.left is None and root.right is None:
                return True
            if root.left and root.right:
                return check_full(root.left) and check_full(root.right)
            return False
        return check_full(self.root)

    def is_complete(self):
        if self.root is None:
            return True
        queue = [self.root]
        flag = False
        while queue:
            node = queue.pop(0)
            if node:
                if flag:
                    return False
                queue.append(node.left)
                queue.append(node.right)
            else:
                flag = True
        return True

    def diameter(self):
        def diameter_of_tree(root):
            if root is None:
                return 0
            left_height = self._height(root.left)
            right_height = self._height(root.right)
            left_diameter = diameter_of_tree(root.left)
            right_diameter = diameter_of_tree(root.right)
            return max(left_height + right_height + 1, max(left_diameter, right_diameter))
        return diameter_of_tree(self.root)

    def find_lca(self, n1, n2):
        def lca(root, n1, n2):
            if root is None:
                return None
            if root.val == n1 or root.val == n2:
                return root
            left_lca = lca(root.left, n1, n2)
            right_lca = lca(root.right, n1, n2)
            if left_lca and right_lca:
                return root
            return left_lca if left_lca else right_lca
        lca_node = lca(self.root, n1, n2)
        return lca_node.val if lca_node else None

    def zigzag_traversal(self):
        if self.root is None:
            return []
        result, current_level, direction = [], [self.root], 1
        while current_level:
            level_result, next_level = [], []
            for node in current_level:
                level_result.append(node.val)
                if node.left:
                    next_level.append(node.left)
                if node.right:
                    next_level.append(node.right)
            result.append(level_result[::direction])
            direction *= -1
            current_level = next_level
        return result

    def find_min(self):
        current = self.root
        while current and current.left:
            current = current.left
        return current.val if current else None

    def find_max(self):
        current = self.root
        while current and current.right:
            current = current.right
        return current.val if current else None

    def sum_of_nodes(self):
        def sum_nodes(root):
            if root is None:
                return 0
            return root.val + sum_nodes(root.left) + sum_nodes(root.right)
        return sum_nodes(self.root)

    def average_of_levels(self):
        if self.root is None:
            return []
        res, current_level = [], [self.root]
        while current_level:
            level_sum, level_count, next_level = 0, 0, []
            for node in current_level:
                level_sum += node.val
                level_count += 1
                if node.left:
                    next_level.append(node.left)
                if node.right:
                    next_level.append(node.right)
            res.append(level_sum / level_count)
            current_level = next_level
        return res

    def are_identical(self, other_tree):
        def check_identical(root1, root2):
            if root1 is None and root2 is None:
                return True
            if root1 is not None and root2 is not None:
                return (root1.val == root2.val and
                        check_identical(root1.left, root2.left) and
                        check_identical(root1.right, root2.right))
            return False
        return check_identical(self.root, other_tree.root)

    def max_depth(self):
        return self._height(self.root)

    def min_depth(self):
        def find_min_depth(root):
            if root is None:
                return 0
            if root.left is None and root.right is None:
                return 1
            if not root.left:
                return find_min_depth(root.right) + 1
            if not root.right:
                return find_min_depth(root.left) + 1
            return min(find_min_depth(root.left), find_min_depth(root.right)) + 1
        return find_min_depth(self.root)

    def vertical_order(self):
        if self.root is None:
            return []
        from collections import deque, defaultdict
        node_queue = deque([(self.root, 0)])
        column_table = defaultdict(list)
        while node_queue:
            node, column = node_queue.popleft()
            if node is not None:
                column_table[column].append(node.val)
                node_queue.append((node.left, column - 1))
                node_queue.append((node.right, column + 1))
        return [column_table[x] for x in sorted(column_table.keys())]

    def right_view(self):
        if self.root is None:
            return []
        right_side, current_level = [], [self.root]
        while current_level:
            right_side.append(current_level[-1].val)
            next_level = []
            for node in current_level:
                if node.left:
                    next_level.append(node.left)
                if node.right:
                    next_level.append(node.right)
            current_level = next_level
        return right_side

    def left_view(self):
        if self.root is None:
            return []
        left_side, current_level = [], [self.root]
        while current_level:
            left_side.append(current_level[0].val)
            next_level = []
            for node in current_level:
                if node.left:
                    next_level.append(node.left)
                if node.right:
                    next_level.append(node.right)
            current_level = next_level
        return left_side

    def top_view(self):
        if self.root is None:
            return []
        from collections import deque, defaultdict
        node_queue = deque([(self.root, 0)])
        column_table = defaultdict(list)
        while node_queue:
            node, column = node_queue.popleft()
            if node is not None:
                if column not in column_table:
                    column_table[column].append(node.val)
                node_queue.append((node.left, column - 1))
                node_queue.append((node.right, column + 1))
        return [column_table[x][0] for x in sorted(column_table.keys())]

    def bottom_view(self):
        if self.root is None:
            return []
        from collections import deque, defaultdict
        node_queue = deque([(self.root, 0)])
        column_table = defaultdict(list)
        while node_queue:
            node, column = node_queue.popleft()
            if node is not None:
                column_table[column].append(node.val)
                node_queue.append((node.left, column - 1))
                node_queue.append((node.right, column + 1))
        return [column_table[x][-1] for x in sorted(column_table.keys())]

    def boundary_traversal(self):
        if self.root is None:
            return []

        def left_boundary(node):
            if node:
                if node.left:
                    yield node.val
                    yield from left_boundary(node.left)
                elif node.right:
                    yield node.val
                    yield from left_boundary(node.right)

        def leaves(node):
            if node:
                yield from leaves(node.left)
                if node.left is None and node.right is None:
                    yield node.val
                yield from leaves(node.right)

        def right_boundary(node):
            if node:
                if node.right:
                    yield from right_boundary(node.right)
                    yield node.val
                elif node.left:
                    yield from right_boundary(node.left)
                    yield node.val

        return [self.root.val] + list(left_boundary(self.root.left)) + list(leaves(self.root)) + list(right_boundary(self.root.right))

    def is_perfect(self):
        def depth(root):
            d = 0
            while root:
                d += 1
                root = root.left
            return d

        def check_perfect(root, d, level=0):
            if root is None:
                return True
            if root.left is None and root.right is None:
                return d == level + 1
            if root.left is None or root.right is None:
                return False
            return check_perfect(root.left, d, level + 1) and check_perfect(root.right, d, level + 1)

        return check_perfect(self.root, depth(self.root))

    def sum_of_left_leaves(self):
        def sum_left_leaves(root, is_left):
            if root is None:
                return 0
            if root.left is None and root.right is None and is_left:
                return root.val
            return sum_left_leaves(root.left, True) + sum_left_leaves(root.right, False)

        return sum_left_leaves(self.root, False)

    def sum_of_right_leaves(self):
        def sum_right_leaves(root, is_right):
            if root is None:
                return 0
            if root.left is None and root.right is None and is_right:
                return root.val
            return sum_right_leaves(root.left, False) + sum_right_leaves(root.right, True)

        return sum_right_leaves(self.root, False)

    def vertical_sum(self):
        if self.root is None:
            return []
        from collections import defaultdict
        column_table = defaultdict(int)
        def vertical_sum_util(node, column):
            if node:
                column_table[column] += node.val
                vertical_sum_util(node.left, column - 1)
                vertical_sum_util(node.right, column + 1)
        vertical_sum_util(self.root, 0)
        return [column_table[x] for x in sorted(column_table.keys())]

    def trim_outside_range(self, min_val, max_val):
        def trim(root, min_val, max_val):
            if root is None:
                return None
            if root.val < min_val:
                return trim(root.right, min_val, max_val)
            if root.val > max_val:
                return trim(root.left, min_val, max_val)
            root.left = trim(root.left, min_val, max_val)
            root.right = trim(root.right, min_val, max_val)
            return root
        self.root = trim(self.root, min_val, max_val)

    def range_sum(self, low, high):
        def range_sum_util(node, low, high):
            if node is None:
                return 0
            if node.val < low:
                return range_sum_util(node.right, low, high)
            if node.val > high:
                return range_sum_util(node.left, low, high)
            return node.val + range_sum_util(node.left, low, high) + range_sum_util(node.right, low, high)
        return range_sum_util(self.root, low, high)

    def find_target_sum(self, k):
        def find_pair(root, target, seen):
            if root is None:
                return False
            if target - root.val in seen:
                return True
            seen.add(root.val)
            return find_pair(root.left, target, seen) or find_pair(root.right, target, seen)
        return find_pair(self.root, k, set())

    def subtree_sum(self):
        def subtree_sum_util(node):
            if node is None:
                return 0, 0
            left_sum, left_count = subtree_sum_util(node.left)
            right_sum, right_count = subtree_sum_util(node.right)
            total_sum = left_sum + right_sum + node.val
            return total_sum, left_count + right_count + 1

        def get_subtree_sum(node):
            if node is None:
                return []
            total_sum, _ = subtree_sum_util(node)
            return [total_sum] + get_subtree_sum(node.left) + get_subtree_sum(node.right)

        return get_subtree_sum(self.root)

    def is_bst(self):
        def is_bst_util(root, left, right):
            if root is None:
                return True
            if not (left < root.val < right):
                return False
            return is_bst_util(root.left, left, root.val) and is_bst_util(root.right, root.val, right)

        return is_bst_util(self.root, float('-inf'), float('inf'))

    def bst_from_preorder(self, preorder):
        if not preorder:
            return None

        root = Node(preorder[0])
        stack = [root]
        for value in preorder[1:]:
            node, child = stack[-1], Node(value)
            while stack and stack[-1].val < value:
                node = stack.pop()
            if node.val < value:
                node.right = child
            else:
                node.left = child
            stack.append(child)

        self.root = root
        return self

    def bst_from_inorder(self, inorder):
        if not inorder:
            return None

        def bst_util(start, end):
            if start > end:
                return None
            mid = (start + end) // 2
            node = Node(inorder[mid])
            node.left = bst_util(start, mid - 1)
            node.right = bst_util(mid + 1, end)
            return node

        self.root = bst_util(0, len(inorder) - 1)
        return self

    def bst_from_postorder(self, postorder):
        if not postorder:
            return None

        def bst_util(low, high):
            if not postorder or postorder[-1] < low or postorder[-1] > high:
                return None
            value = postorder.pop()
            node = Node(value)
            node.right = bst_util(value, high)
            node.left = bst_util(low, value)
            return node

        self.root = bst_util(float('-inf'), float('inf'))
        return self

    def bst_from_level_order(self, level_order):
        if not level_order:
            return None

        root = Node(level_order[0])
        queue = [(root, float('-inf'), float('inf'))]
        i = 1
        while queue:
            node, low, high = queue.pop(0)
            if i < len(level_order) and low < level_order[i] < node.val:
                node.left = Node(level_order[i])
                queue.append((node.left, low, node.val))
                i += 1
            if i < len(level_order) and node.val < level_order[i] < high:
                node.right = Node(level_order[i])
                queue.append((node.right, node.val, high))
                i += 1

        self.root = root
        return self

    def bst_to_sorted_list(self):
        return self.inorder()

    def sorted_list_to_bst(self, sorted_list):
        if not sorted_list:
            return None

        def convert_to_bst(left, right):
            if left > right:
                return None
            mid = (left + right) // 2
            node = Node(sorted_list[mid])
            node.left = convert_to_bst(left, mid - 1)
            node.right = convert_to_bst(mid + 1, right)
            return node

        self.root = convert_to_bst(0, len(sorted_list) - 1)
        return self

    def count_bst_nodes_in_range(self, low, high):
        def count_nodes(root, low, high):
            if root is None:
                return 0
            if low <= root.val <= high:
                return 1 + count_nodes(root.left, low, high) + count_nodes(root.right, low, high)
            elif root.val < low:
                return count_nodes(root.right, low, high)
            else:
                return count_nodes(root.left, low, high)
        return count_nodes(self.root, low, high)

    def sum_bst_nodes_in_range(self, low, high):
        def sum_nodes(root, low, high):
            if root is None:
                return 0
            if low <= root.val <= high:
                return root.val + sum_nodes(root.left, low, high) + sum_nodes(root.right, low, high)
            elif root.val < low:
                return sum_nodes(root.right, low, high)
            else:
                return sum_nodes(root.left, low, high)
        return sum_nodes(self.root, low, high)

    def bst_kth_smallest(self, k):
        def kth_smallest_util(root, k):
            stack = []
            while True:
                while root:
                    stack.append(root)
                    root = root.left
                root = stack.pop()
                k -= 1
                if k == 0:
                    return root.val
                root = root.right
        return kth_smallest_util(self.root, k)

    def bst_kth_largest(self, k):
        def kth_largest_util(root, k):
            stack = []
            while True:
                while root:
                    stack.append(root)
                    root = root.right
                root = stack.pop()
                k -= 1
                if k == 0:
                    return root.val
                root = root.left
        return kth_largest_util(self.root, k)

    def merge_two_bsts(self, bst1, bst2):
        def merge_trees(root1, root2):
            if root1 is None:
                return root2
            if root2 is None:
                return root1
            root1.val += root2.val
            root1.left = merge_trees(root1.left, root2.left)
            root1.right = merge_trees(root1.right, root2.right)
            return root1

        merged_tree = BinaryTree()
        merged_tree.root = merge_trees(bst1.root, bst2.root)
        return merged_tree

    def is_bst_valid(self):
        def validate_bst(root, low, high):
            if root is None:
                return True
            if not (low < root.val < high):
                return False
            return validate_bst(root.left, low, root.val) and validate_bst(root.right, root.val, high)

        return validate_bst(self.root, float('-inf'), float('inf'))

    def bst_nodes_at_distance_k(self, k):
        def nodes_at_distance_k(root, k, current_distance):
            if root is None:
                return []
            if current_distance == k:
                return [root.val]
            return nodes_at_distance_k(root.left, k, current_distance + 1) + nodes_at_distance_k(root.right, k, current_distance + 1)

        return nodes_at_distance_k(self.root, k, 0)

    def bst_nodes_at_level(self, level):
        def nodes_at_level(root, current_level, target_level):
            if root is None:
                return []
            if current_level == target_level:
                return [root.val]
            return nodes_at_level(root.left, current_level + 1, target_level) + nodes_at_level(root.right, current_level + 1, target_level)

        return nodes_at_level(self.root, 0, level)

    def bst_find_successor(self, key):
        def find_successor(root, key):
            successor = None
            while root:
                if key < root.val:
                    successor = root
                    root = root.left
                elif key > root.val:
                    root = root.right
                else:
                    if root.right:
                        successor = root.right
                        while successor.left:
                            successor = successor.left
                    break
            return successor

        successor_node = find_successor(self.root, key)
        return successor_node.val if successor_node else None

    def bst_find_predecessor(self, key):
        def find_predecessor(root, key):
            predecessor = None
            while root:
                if key > root.val:
                    predecessor = root
                    root = root.right
                elif key < root.val:
                    root = root.left
                else:
                    if root.left:
                        predecessor = root.left
                        while predecessor.right:
                            predecessor = predecessor.right
                    break
            return predecessor

        predecessor_node = find_predecessor(self.root, key)
        return predecessor_node.val if predecessor_node else None

    def bst_get_all_paths(self):
        def all_paths(root, path, paths):
            if root is None:
                return
            path.append(root.val)
            if root.left is None and root.right is None:
                paths.append(list(path))
            else:
                all_paths(root.left, path, paths)
                all_paths(root.right, path, paths)
            path.pop()

        paths = []
        all_paths(self.root, [], paths)
        return paths

    def bst_mirror(self):
        def mirror(root):
            if root is None:
                return None
            root.left, root.right = mirror(root.right), mirror(root.left)
            return root

        self.root = mirror(self.root)
        return self

    def bst_is_mirror(self, other_tree):
        def is_mirror_tree(root1, root2):
            if root1 is None and root2 is None:
                return True
            if root1 is None or root2 is None:
                return False
            return (root1.val == root2.val and
                    is_mirror_tree(root1.left, root2.right) and
                    is_mirror_tree(root1.right, root2.left))

        return is_mirror_tree(self.root, other_tree.root)

    def bst_find_distance(self, key1, key2):
        def find_lca_distance(root, key1, key2):
            if root is None:
                return None
            if root.val == key1 or root.val == key2:
                return root
            left_lca = find_lca_distance(root.left, key1, key2)
            right_lca = find_lca_distance(root.right, key1, key2)
            if left_lca and right_lca:
                return root
            return left_lca if left_lca else right_lca

        def find_distance_from_lca(root, key, distance):
            if root is None:
                return -1
            if root.val == key:
                return distance
            left_distance = find_distance_from_lca(root.left, key, distance + 1)
            if left_distance != -1:
                return left_distance
            return find_distance_from_lca(root.right, key, distance + 1)

        lca = find_lca_distance(self.root, key1, key2)
        if lca is None:
            return -1
        distance1 = find_distance_from_lca(lca, key1, 0)
        distance2 = find_distance_from_lca(lca, key2, 0)
        return distance1 + distance2 if distance1 != -1 and distance2 != -1 else -1

    def bst_max_path_sum(self):
        def max_path_sum_util(root):
            if root is None:
                return 0
            left = max(0, max_path_sum_util(root.left))
            right = max(0, max_path_sum_util(root.right))
            return root.val + max(left, right)

        return max_path_sum_util(self.root)

    def bst_inorder_successor(self, key):
        def inorder_successor(root, key):
            successor = None
            while root:
                if key < root.val:
                    successor = root
                    root = root.left
                else:
                    root = root.right
            return successor

        return inorder_successor(self.root, key).val

    def bst_inorder_predecessor(self, key):
        def inorder_predecessor(root, key):
            predecessor = None
            while root:
                if key > root.val:
                    predecessor = root
                    root = root.right
                else:
                    root = root.left
            return predecessor

        return inorder_predecessor(self.root, key).val

    def bst_find_min_max(self):
        def find_min(root):
            while root.left:
                root = root.left
            return root.val

        def find_max(root):
            while root.right:
                root = root.right
            return root.val

        return find_min(self.root), find_max(self.root)

    def bst_path_to_node(self, key):
        def find_path(root, key, path):
            if root is None:
                return False
            path.append(root.val)
            if root.val == key:
                return True
            if (root.left and find_path(root.left, key, path)) or (root.right and find_path(root.right, key, path)):
                return True
            path.pop()
            return False

        path = []
        find_path(self.root, key, path)
        return path

    def bst_print_all_paths(self):
        def print_all_paths(root, path):
            if root is None:
                return
            path.append(root.val)
            if root.left is None and root.right is None:
                print(" -> ".join(map(str, path)))
            else:
                print_all_paths(root.left, path)
                print_all_paths(root.right, path)
            path.pop()

        print_all_paths(self.root, [])

    def bst_longest_path_sum(self):
        def longest_path_sum_util(root):
            if root is None:
                return 0, 0
            left_length, left_sum = longest_path_sum_util(root.left)
            right_length, right_sum = longest_path_sum_util(root.right)
            if left_length > right_length:
                return left_length + 1, left_sum + root.val
            elif left_length < right_length:
                return right_length + 1, right_sum + root.val
            else:
                return left_length + 1, max(left_sum, right_sum) + root.val

        return longest_path_sum_util(self.root)[1]

    def bst_find_path_sum(self, sum):
        def find_path_sum(root, sum, current_sum, path):
            if root is None:
                return []
            current_sum += root.val
            path.append(root.val)
            if current_sum == sum:
                return [list(path)]
            paths = find_path_sum(root.left, sum, current_sum, path) + find_path_sum(root.right, sum, current_sum, path)
            path.pop()
            return paths

        return find_path_sum(self.root, sum, 0, [])

    def bst_convert_to_greater_tree(self):
        def convert_to_greater_tree(root, acc_sum):
            if root is None:
                return acc_sum
            acc_sum = convert_to_greater_tree(root.right, acc_sum)
            acc_sum += root.val
            root.val = acc_sum
            acc_sum = convert_to_greater_tree(root.left, acc_sum)
            return acc_sum

        convert_to_greater_tree(self.root, 0)
        return self
