class Node:
    def __init__(self, key: int):
        self.key = key
        self.left = None
        self.right = None


class BinarySearchTree:
    def __init__(self):
        self.root = None

    def insert(self, key: int) -> None:
        if self.root is None:
            self.root = Node(key)
        else:
            self._insert(self.root, key)

    def _insert(self, node: Node, key: int) -> None:
        if key < node.key:
            if node.left is None:
                node.left = Node(key)
            else:
                self._insert(node.left, key)
        else:
            if node.right is None:
                node.right = Node(key)
            else:
                self._insert(node.right, key)

    def search(self, key: int) -> Node:
        return self._search(self.root, key)

    def _search(self, node: Node, key: int) -> Node:
        if node is None or node.key == key:
            return node
        if key < node.key:
            return self._search(node.left, key)
        return self._search(node.right, key)

    def delete(self, key: int) -> None:
        self.root = self._delete(self.root, key)

    def _delete(self, node: Node, key: int) -> Node:
        if node is None:
            return node
        if key < node.key:
            node.left = self._delete(node.left, key)
        elif key > node.key:
            node.right = self._delete(node.right, key)
        else:
            if node.left is None:
                return node.right
            elif node.right is None:
                return node.left
            temp = self._min_value_node(node.right)
            node.key = temp.key
            node.right = self._delete(node.right, temp.key)
        return node

    def _min_value_node(self, node: Node) -> Node:
        current = node
        while current.left is not None:
            current = current.left
        return current

    def inorder(self) -> list:
        return self._inorder(self.root)

    def _inorder(self, node: Node) -> list:
        res = []
        if node:
            res = self._inorder(node.left)
            res.append(node.key)
            res = res + self._inorder(node.right)
        return res

    def preorder(self) -> list:
        return self._preorder(self.root)

    def _preorder(self, node: Node) -> list:
        res = []
        if node:
            res.append(node.key)
            res = res + self._preorder(node.left)
            res = res + self._preorder(node.right)
        return res

    def postorder(self) -> list:
        return self._postorder(self.root)

    def _postorder(self, node: Node) -> list:
        res = []
        if node:
            res = self._postorder(node.left)
            res = res + self._postorder(node.right)
            res.append(node.key)
        return res

    def height(self) -> int:
        return self._height(self.root)

    def _height(self, node: Node) -> int:
        if node is None:
            return -1
        left_height = self._height(node.left)
        right_height = self._height(node.right)
        return 1 + max(left_height, right_height)

    def is_balanced(self) -> bool:
        def check(node: Node) -> tuple[int, bool]:
            if not node:
                return 0, True
            lh, lb = check(node.left)
            rh, rb = check(node.right)
            h = max(lh, rh) + 1
            b = lb and rb and abs(lh - rh) <= 1
            return h, b

        return check(self.root)[1]

    def level_order(self) -> list:
        if not self.root:
            return []
        queue, res = [self.root], []
        while queue:
            current = queue.pop(0)
            res.append(current.key)
            if current.left:
                queue.append(current.left)
            if current.right:
                queue.append(current.right)
        return res

    def min_value(self) -> int:
        if self.root is None:
            return None
        return self._min_value_node(self.root).key

    def max_value(self) -> int:
        if self.root is None:
            return None
        current = self.root
        while current.right:
            current = current.right
        return current.key

    def find_lca(self, n1: int, n2: int) -> Node:
        return self._find_lca(self.root, n1, n2)

    def _find_lca(self, node: Node, n1: int, n2: int) -> Node:
        if node is None:
            return None
        if node.key > n1 and node.key > n2:
            return self._find_lca(node.left, n1, n2)
        if node.key < n1 and node.key < n2:
            return self._find_lca(node.right, n1, n2)
        return node

    def distance(self, n1: int, n2: int) -> int:
        lca = self.find_lca(n1, n2)
        return self._distance_from_lca(lca, n1) + self._distance_from_lca(lca, n2)

    def _distance_from_lca(self, node: Node, key: int) -> int:
        if node is None:
            return -1
        if node.key == key:
            return 0
        elif node.key > key:
            return 1 + self._distance_from_lca(node.left, key)
        return 1 + self._distance_from_lca(node.right, key)

    def kth_smallest(self, k: int) -> int:
        stack = []
        current = self.root
        while True:
            while current:
                stack.append(current)
                current = current.left
            current = stack.pop()
            k -= 1
            if k == 0:
                return current.key
            current = current.right

    def kth_largest(self, k: int) -> int:
        stack = []
        current = self.root
        while True:
            while current:
                stack.append(current)
                current = current.right
            current = stack.pop()
            k -= 1
            if k == 0:
                return current.key
            current = current.left

    def sum(self) -> int:
        return self._sum(self.root)

    def _sum(self, node: Node) -> int:
        if node is None:
            return 0
        return node.key + self._sum(node.left) + self._sum(node.right)

    def count(self) -> int:
        return self._count(self.root)

    def _count(self, node: Node) -> int:
        if node is None:
            return 0
        return 1 + self._count(node.left) + self._count(node.right)

    def is_symmetric(self) -> bool:
        def is_mirror(t1: Node, t2: Node) -> bool:
            if not t1 and not t2:
                return True
            if not t1 or not t2:
                return False
            return (t1.key == t2.key
                    and is_mirror(t1.left, t2.right)
                    and is_mirror(t1.right, t2.left))

        return is_mirror(self.root, self.root)

    def find_path(self, key: int) -> list:
        path = []
        if not self._find_path(self.root, path, key):
            return None
        return path

    def _find_path(self, node: Node, path: list, key: int) -> bool:
        if node is None:
            return False
        path.append(node.key)
        if node.key == key:
            return True
        if ((node.left and self._find_path(node.left, path, key)) or
                (node.right and self._find_path(node.right, path, key))):
            return True
        path.pop()
        return False

    def nodes_at_distance_k(self, k: int) -> list:
        result = []
        self._nodes_at_distance_k(self.root, k, result)
        return result

    def _nodes_at_distance_k(self, node: Node, k: int, result: list) -> None:
        if node is None:
            return
        if k == 0:
            result.append(node.key)
            return
        self._nodes_at_distance_k(node.left, k-1, result)
        self._nodes_at_distance_k(node.right, k-1, result)

    def mirror(self) -> None:
        def mirror_tree(node: Node) -> Node:
            if node is None:
                return None
            node.left, node.right = mirror_tree(node.right), mirror_tree(node.left)
            return node

        self.root = mirror_tree(self.root)

    def is_mirror(self, other_tree: 'BinarySearchTree') -> bool:
        def is_mirror_tree(n1: Node, n2: Node) -> bool:
            if not n1 and not n2:
                return True
            if not n1 or not n2:
                return False
            return (n1.key == n2.key
                    and is_mirror_tree(n1.left, n2.right)
                    and is_mirror_tree(n1.right, n2.left))

        return is_mirror_tree(self.root, other_tree.root)

    def max_path_sum(self) -> int:
        def max_sum_util(node: Node) -> int:
            if node is None:
                return 0
            left = max(0, max_sum_util(node.left))
            right = max(0, max_sum_util(node.right))
            max_sum_util.max_sum = max(max_sum_util.max_sum, left + right + node.key)
            return node.key + max(left, right)

        max_sum_util.max_sum = float('-inf')
        max_sum_util(self.root)
        return max_sum_util.max_sum

    def invert(self) -> None:
        def invert_tree(node: Node) -> Node:
            if node is None:
                return None
            node.left, node.right = invert_tree(node.right), invert_tree(node.left)
            return node

        self.root = invert_tree(self.root)

    def path_sum(self, sum: int) -> list:
        def path_sum_util(node: Node, remaining_sum: int, path: list, result: list) -> None:
            if node is None:
                return
            path.append(node.key)
            remaining_sum -= node.key
            if remaining_sum == 0 and node.left is None and node.right is None:
                result.append(list(path))
            else:
                path_sum_util(node.left, remaining_sum, path, result)
                path_sum_util(node.right, remaining_sum, path, result)
            path.pop()

        result = []
        path_sum_util(self.root, sum, [], result)
        return result

    def lowest_common_ancestor(self, p: int, q: int) -> Node:
        def lca_util(node: Node, p: int, q: int) -> Node:
            if node is None:
                return None
            if node.key > p and node.key > q:
                return lca_util(node.left, p, q)
            if node.key < p and node.key < q:
                return lca_util(node.right, p, q)
            return node

        return lca_util(self.root, p, q)

    def right_side_view(self) -> list:
        result = []
        if not self.root:
            return result
        queue = [self.root]
        while queue:
            level_size = len(queue)
            for i in range(level_size):
                node = queue.pop(0)
                if i == level_size - 1:
                    result.append(node.key)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return result

    def left_side_view(self) -> list:
        result = []
        if not self.root:
            return result
        queue = [self.root]
        while queue:
            level_size = len(queue)
            for i in range(level_size):
                node = queue.pop(0)
                if i == 0:
                    result.append(node.key)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return result

    def vertical_order(self) -> list:
        from collections import defaultdict, deque
        if not self.root:
            return []
        column_table = defaultdict(list)
        queue = deque([(self.root, 0)])
        while queue:
            node, column = queue.popleft()
            column_table[column].append(node.key)
            if node.left:
                queue.append((node.left, column - 1))
            if node.right:
                queue.append((node.right, column + 1))
        return [column_table[x] for x in sorted(column_table.keys())]

    def zigzag_level_order(self) -> list:
        result = []
        if not self.root:
            return result
        queue = [self.root]
        left_to_right = True
        while queue:
            level_size = len(queue)
            level_nodes = []
            for _ in range(level_size):
                node = queue.pop(0)
                if left_to_right:
                    level_nodes.append(node.key)
                else:
                    level_nodes.insert(0, node.key)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            result.append(level_nodes)
            left_to_right = not left_to_right
        return result

    def sum_of_left_leaves(self) -> int:
        def left_leaves_sum(node: Node, is_left: bool) -> int:
            if node is None:
                return 0
            if node.left is None and node.right is None and is_left:
                return node.key
            return left_leaves_sum(node.left, True) + left_leaves_sum(node.right, False)

        return left_leaves_sum(self.root, False)

    def range_sum_bst(self, low: int, high: int) -> int:
        def range_sum(node: Node, low: int, high: int) -> int:
            if node is None:
                return 0
            if node.key < low:
                return range_sum(node.right, low, high)
            if node.key > high:
                return range_sum(node.left, low, high)
            return node.key + range_sum(node.left, low, high) + range_sum(node.right, low, high)

        return range_sum(self.root, low, high)

    def diameter(self) -> int:
        def diameter_util(node: Node) -> tuple[int, int]:
            if node is None:
                return 0, 0
            left_height, left_diameter = diameter_util(node.left)
            right_height, right_diameter = diameter_util(node.right)
            current_height = 1 + max(left_height, right_height)
            current_diameter = max(left_diameter, right_diameter, left_height + right_height + 1)
            return current_height, current_diameter

        return diameter_util(self.root)[1]

    def max_depth(self) -> int:
        return self.height()

    def min_depth(self) -> int:
        def min_depth_util(node: Node) -> int:
            if node is None:
                return 0
            if not node.left:
                return 1 + min_depth_util(node.right)
            if not node.right:
                return 1 + min_depth_util(node.left)
            return 1 + min(min_depth_util(node.left), min_depth_util(node.right))

        return min_depth_util(self.root)

    def path_sum_root_to_leaf(self, sum: int) -> bool:
        def path_sum_util(node: Node, remaining_sum: int) -> bool:
            if node is None:
                return False
            remaining_sum -= node.key
            if not node.left and not node.right:
                return remaining_sum == 0
            return path_sum_util(node.left, remaining_sum) or path_sum_util(node.right, remaining_sum)

        return path_sum_util(self.root, sum)

    def has_path_sum(self, sum: int) -> bool:
        return self.path_sum_root_to_leaf(sum)

    def leaf_nodes(self) -> list:
        def leaf_nodes_util(node: Node, leaves: list) -> None:
            if node is None:
                return
            if not node.left and not node.right:
                leaves.append(node.key)
            leaf_nodes_util(node.left, leaves)
            leaf_nodes_util(node.right, leaves)

        leaves = []
        leaf_nodes_util(self.root, leaves)
        return leaves

    def boundary_traversal(self) -> list:
        def left_boundary(node: Node) -> list:
            if node is None or (node.left is None and node.right is None):
                return []
            if node.left:
                return [node.key] + left_boundary(node.left)
            return [node.key] + left_boundary(node.right)

        def right_boundary(node: Node) -> list:
            if node is None or (node.left is None and node.right is None):
                return []
            if node.right:
                return right_boundary(node.right) + [node.key]
            return right_boundary(node.left) + [node.key]

        def leaves(node: Node) -> list:
            if node is None:
                return []
            if node.left is None and node.right is None:
                return [node.key]
            return leaves(node.left) + leaves(node.right)

        if not self.root:
            return []
        return [self.root.key] + left_boundary(self.root.left) + leaves(self.root.left) + leaves(self.root.right) + right_boundary(self.root.right)

    def top_view(self) -> list:
        if not self.root:
            return []
        top_view_map = {}
        queue = [(self.root, 0)]
        while queue:
            node, hd = queue.pop(0)
            if hd not in top_view_map:
                top_view_map[hd] = node.key
            if node.left:
                queue.append((node.left, hd - 1))
            if node.right:
                queue.append((node.right, hd + 1))
        return [top_view_map[key] for key in sorted(top_view_map.keys())]

    def bottom_view(self) -> list:
        if not self.root:
            return []
        bottom_view_map = {}
        queue = [(self.root, 0)]
        while queue:
            node, hd = queue.pop(0)
            bottom_view_map[hd] = node.key
            if node.left:
                queue.append((node.left, hd - 1))
            if node.right:
                queue.append((node.right, hd + 1))
        return [bottom_view_map[key] for key in sorted(bottom_view_map.keys())]

    def diagonal_traversal(self) -> list:
        if not self.root:
            return []
        diagonal_map = {}
        queue = [(self.root, 0)]
        while queue:
            node, d = queue.pop(0)
            if d in diagonal_map:
                diagonal_map[d].append(node.key)
            else:
                diagonal_map[d] = [node.key]
            if node.left:
                queue.append((node.left, d + 1))
            if node.right:
                queue.append((node.right, d))
        result = []
        for k in sorted(diagonal_map.keys()):
            result += diagonal_map[k]
        return result

    def construct_from_preorder_inorder(self, preorder: list, inorder: list) -> Node:
        if not preorder or not inorder:
            return None
        root = Node(preorder[0])
        mid = inorder.index(preorder[0])
        root.left = self.construct_from_preorder_inorder(preorder[1:mid+1], inorder[:mid])
        root.right = self.construct_from_preorder_inorder(preorder[mid+1:], inorder[mid+1:])
        return root

    def construct_from_inorder_postorder(self, inorder: list, postorder: list) -> Node:
        if not inorder or not postorder:
            return None
        root = Node(postorder.pop())
        mid = inorder.index(root.key)
        root.right = self.construct_from_inorder_postorder(inorder[mid+1:], postorder)
        root.left = self.construct_from_inorder_postorder(inorder[:mid], postorder)
        return root

    def construct_from_preorder_postorder(self, preorder: list, postorder: list) -> Node:
        if not preorder or not postorder:
            return None
        root = Node(preorder.pop(0))
        if root.key != postorder.pop():
            i = postorder.index(preorder[0])
            root.left = self.construct_from_preorder_postorder(preorder, postorder[:i+1])
            root.right = self.construct_from_preorder_postorder(preorder, postorder[i+1:])
        return root

    def path_sum_all(self, sum: int) -> list:
        def path_sum_all_util(node: Node, remaining_sum: int, path: list, result: list) -> None:
            if node is None:
                return
            path.append(node.key)
            remaining_sum -= node.key
            if remaining_sum == 0 and not node.left and not node.right:
                result.append(list(path))
            else:
                path_sum_all_util(node.left, remaining_sum, path, result)
                path_sum_all_util(node.right, remaining_sum, path, result)
            path.pop()

        result = []
        path_sum_all_util(self.root, sum, [], result)
        return result

    def sum_of_nodes(self) -> int:
        return self.sum()

    def average_of_levels(self) -> list:
        if not self.root:
            return []
        queue = [self.root]
        result = []
        while queue:
            level_sum = 0
            level_count = len(queue)
            for _ in range(level_count):
                node = queue.pop(0)
                level_sum += node.key
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            result.append(level_sum / level_count)
        return result

    def max_level_sum(self) -> int:
        if not self.root:
            return 0
        max_sum = float('-inf')
        queue = [self.root]
        while queue:
            level_sum = 0
            for _ in range(len(queue)):
                node = queue.pop(0)
                level_sum += node.key
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            max_sum = max(max_sum, level_sum)
        return max_sum

    def count_good_nodes(self) -> int:
        def count_good_nodes_util(node: Node, max_value: int) -> int:
            if node is None:
                return 0
            total = 1 if node.key >= max_value else 0
            max_value = max(max_value, node.key)
            total += count_good_nodes_util(node.left, max_value)
            total += count_good_nodes_util(node.right, max_value)
            return total

        return count_good_nodes_util(self.root, float('-inf'))

    def is_complete(self) -> bool:
        if not self.root:
            return True
        queue = [self.root]
        reached_end = False
        while queue:
            node = queue.pop(0)
            if not node:
                reached_end = True
            else:
                if reached_end:
                    return False
                queue.append(node.left)
                queue.append(node.right)
        return True

    def is_perfect(self) -> bool:
        def check_perfect(node: Node) -> tuple[int, bool]:
            if not node:
                return 0, True
            lh, lp = check_perfect(node.left)
            rh, rp = check_perfect(node.right)
            h = max(lh, rh) + 1
            p = lp and rp and lh == rh
            return h, p

        return check_perfect(self.root)[1]

    def all_root_to_leaf_paths(self) -> list:
        def all_paths_util(node: Node, path: list, result: list) -> None:
            if node is None:
                return
            path.append(node.key)
            if not node.left and not node.right:
                result.append(list(path))
            else:
                all_paths_util(node.left, path, result)
                all_paths_util(node.right, path, result)
            path.pop()

        result = []
        all_paths_util(self.root, [], result)
        return result

    def prune(self, sum: int) -> None:
        def prune_util(node: Node, sum: int) -> Node:
            if node is None:
                return None
            node.left = prune_util(node.left, sum - node.key)
            node.right = prune_util(node.right, sum - node.key)
            if not node.left and not node.right and node.key < sum:
                return None
            return node

        self.root = prune_util(self.root, sum)

    def sum_of_path_numbers(self) -> int:
        def sum_path_util(node: Node, current_sum: int) -> int:
            if node is None:
                return 0
            current_sum = current_sum * 10 + node.key
            if not node.left and not node.right:
                return current_sum
            return sum_path_util(node.left, current_sum) + sum_path_util(node.right, current_sum)

        return sum_path_util(self.root, 0)

    def serialize(self) -> str:
        def serialize_util(node: Node) -> str:
            if node is None:
                return "None,"
            return str(node.key) + "," + serialize_util(node.left) + serialize_util(node.right)

        return serialize_util(self.root)

    def deserialize(self, data: str) -> None:
        def deserialize_util(data_list: list) -> Node:
            if data_list[0] == "None":
                data_list.pop(0)
                return None
            node = Node(int(data_list[0]))
            data_list.pop(0)
            node.left = deserialize_util(data_list)
            node.right = deserialize_util(data_list)
            return node

        data_list = data.split(',')
        self.root = deserialize_util(data_list[:-1])

    def count_leaves(self) -> int:
        def count_leaves_util(node: Node) -> int:
            if node is None:
                return 0
            if node.left is None and node.right is None:
                return 1
            return count_leaves_util(node.left) + count_leaves_util(node.right)

        return count_leaves_util(self.root)

    def count_full_nodes(self) -> int:
        def count_full_nodes_util(node: Node) -> int:
            if node is None:
                return 0
            count = 1 if node.left and node.right else 0
            count += count_full_nodes_util(node.left)
            count += count_full_nodes_util(node.right)
            return count

        return count_full_nodes_util(self.root)

    def count_single_child_nodes(self) -> int:
        def count_single_child_nodes_util(node: Node) -> int:
            if node is None:
                return 0
            count = 1 if (node.left and not node.right) or (not node.left and node.right) else 0
            count += count_single_child_nodes_util(node.left)
            count += count_single_child_nodes_util(node.right)
            return count

        return count_single_child_nodes_util(self.root)
