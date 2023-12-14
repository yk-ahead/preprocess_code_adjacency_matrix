import re

import numpy as np
from constants import PAD_WORD, SPLITTED_WORD, SUBTOKEN_WORD
from DFG import DFG_java


class anyNode(object):
    def __init__(self, data, node_type, token, kids=[], siblings=[]):
        self.data = data
        self.type = node_type
        self.token = token
        self.kids = kids
        self.siblings = siblings  # only the siblings on the "right" side


def split_camelcase(camel_case_identifier):
    # def split_camelcase(camel_case_identifier: str) -> List[str]:
    """
    Split camelCase identifiers.
    come from code transformer
    """
    if not len(camel_case_identifier):
        return []
    # split into words based on adjacent cases being the same
    result = []
    current = str(camel_case_identifier[0])
    prev_upper = camel_case_identifier[0].isupper()
    prev_digit = camel_case_identifier[0].isdigit()
    prev_special = not camel_case_identifier[0].isalnum()
    for c in camel_case_identifier[1:]:
        upper = c.isupper()
        digit = c.isdigit()
        special = not c.isalnum()
        new_upper_word = upper and not prev_upper
        new_digit_word = digit and not prev_digit
        new_special_word = special and not prev_special
        if new_digit_word or new_upper_word or new_special_word:
            result.append(current)
            current = c
        elif not upper and prev_upper and len(current) > 1:
            result.append(current[:-1])
            current = current[-1] + c
        elif not digit and prev_digit:
            result.append(current)
            current = c
        elif not special and prev_special:
            result.append(current)
            current = c
        else:
            current += c
        prev_digit = digit
        prev_upper = upper
        prev_special = special
    result.append(current)

    return result


def cap(line):
    # 判断一个字符串中是否包含大写字母
    # flag = False
    for x in line:
        if x.isupper():
            return True
    return False


def split_identifier_into_parts(identifier):
    """
    Split a single identifier into parts on snake_case and camelCase
    come from code transformer
    """
    snake_case = identifier.split("_")
    identifier_parts = []
    for i in range(len(snake_case)):
        part = snake_case[i]

        if len(part) > 0:
            if not cap(part):
                identifier_parts.append(part)
            else:
                # identifier_parts.extend(split_camelcase(part))
                # identifier_parts.append(split_camelcase(part))
                split_parts = split_camelcase(part)
                lower_split_parts = [s.lower().strip() for s in split_parts]
                # identifier_parts.append(s.lower() for s in split_camelcase(part))
                identifier_parts.extend(lower_split_parts)

    return identifier_parts


def get_token(node, lower=False):
    """
    Get the token of an ast node,
    the token of a leaf node is its text in code,
    the token of a non-leaf node is its ast type.
    """
    if len(node.children) == 0:
        token = re.sub(r"\s", "", str(node.text, "utf-8"))
        if not node.is_named:
            token = node.type
    else:
        token = node.type
    if lower:
        return token.lower()
    if token == "":
        token = PAD_WORD
    return token


def get_children(node):
    """Get all children of an ast node."""
    return node.children


def get_siblings(node):
    """Get all siblings on the 'right' of an ast node."""
    siblings = []
    while node.next_sibling:
        node = node.next_sibling
        siblings.append(node)
    return siblings


def ast2anytree(tree, split=False):
    """
    Turn ast to anytree.  Require package 'anytree'.
    If split is True, split the identifier into parts
    according to camelCase and snake_case.

    Args:
        tree (tree_sitter.Tree): The root node of the giving ast tree.

    Returns:
        newtree (AnyNode): The root node of the generated anytree.
    """
    from anytree import AnyNode

    def create_tree(node, parent=None):
        children = node.children
        token = get_token(node)
        if len(children) == 0:
            if split:
                subnodes = [
                    AnyNode(token=subtoken, type=SUBTOKEN_WORD)
                    for subtoken in split_identifier_into_parts(token)
                ]
                if len(subnodes) > 1:
                    newnode = AnyNode(
                        token=SPLITTED_WORD,
                        type=node.type,
                        parent=parent,
                        start_point=node.start_point,
                        end_point=node.end_point,
                    )
                    for anyNode in subnodes:
                        anyNode.parent = newnode
                else:
                    newnode = AnyNode(
                        token=token.lower(),
                        type=node.type,
                        parent=parent,
                        start_point=node.start_point,
                        end_point=node.end_point,
                    )
            else:
                newnode = AnyNode(
                    token=token.lower(),
                    type=node.type,
                    parent=parent,
                    start_point=node.start_point,
                    end_point=node.end_point,
                )
        else:
            newnode = AnyNode(
                token=token.lower(),
                type=node.type,
                parent=parent,
                start_point=node.start_point,
                end_point=node.end_point,
            )
            for child in children:
                create_tree(child, parent=newnode)
        return newnode

    new_tree = create_tree(tree.root_node)
    return new_tree


def walk(root):
    """
    Tranversing the AST, get the AST nodes in the order of the
    code tokens in the original code snippet

    :param tree: tree_sitter.Tree Object
    :return: list of AST nodes
    """
    ast = []
    stack = [root]
    while stack:
        current = stack.pop()
        children = []
        ins = len(stack)
        for c in current.children:
            children += [c]
            stack.insert(ins, c)
        ast.append(current)
    return ast


def get_flatten_ast(tree, split=False):
    any_tree = ast2anytree(tree, split)
    flatten_ast = [anyNode for anyNode in walk(any_tree) if anyNode.type != "comment"]
    return flatten_ast


def tree_to_token_index(root_node):
    if (len(root_node.children) == 0 or root_node.type == "string") and root_node.type != "comment":
        return [(root_node.start_point, root_node.end_point)]
    else:
        code_tokens = []
        for child in root_node.children:
            code_tokens += tree_to_token_index(child)
        return code_tokens


def get_index2code(code, flatten_ast):
    """Get index2code for extracting DFG."""
    index2code = {}
    for idx, node in enumerate(flatten_ast):
        if (
            (len(node.children) == 0 and node.type is not SUBTOKEN_WORD)
            or (node.token == SPLITTED_WORD)
            or node.type == "string"
        ):
            token_pos = (node.start_point, node.end_point)
            code_token = pos_to_code_token(token_pos, code)
            index2code[token_pos] = (idx, code_token)
    return index2code


def pos_to_code_token(pos, code):
    start_point = pos[0]
    end_point = pos[1]
    if start_point[0] == end_point[0]:
        s = code[start_point[0]][start_point[1] : end_point[1]]
    else:
        s = ""
        s += code[start_point[0]][start_point[1] :]
        for i in range(start_point[0] + 1, end_point[0]):
            s += code[i]
        s += code[end_point[0]][: end_point[1]]
    return s


def get_ast_matrix(all_nodes, max_len=512):
    """
    Get the adjacency matrix from the AST
    according to the parent-children relationship

    :param all_nodes: all AST nodes, the terminal nodes have been splited
    :return:
    matrix: AST adjacency matrix
    """

    length = len(all_nodes)
    matrix = (
        np.zeros([length, length], dtype=np.bool)
        if length <= max_len
        else np.zeros([max_len, max_len], dtype=np.bool)
    )

    for i, node in enumerate(all_nodes):
        if i >= max_len:
            continue
        matrix[i][i] = 1
        for child in node.children:
            j = all_nodes.index(child)
            if j < max_len:
                matrix[i][j] = 1
                matrix[j][i] = 1

    return matrix


def get_sib_matrix(all_nodes, max_len=512):
    """
    Get the adjacency matrix from the AST
    according to the sibling relationship

    :param all_nodes: all AST nodes, the terminal nodes have been splited
    :return:
    matrix: AST adjacency matrix
    """

    length = len(all_nodes)
    matrix = (
        np.zeros([length, length], dtype=np.bool)
        if length <= max_len
        else np.zeros([max_len, max_len], dtype=np.bool)
    )

    for i, node in enumerate(all_nodes):
        if i >= max_len:
            continue
        for child in node.siblings:
            j = all_nodes.index(child)
            if j < max_len:
                matrix[i][j] = 1
                matrix[j][i] = 1

    return matrix


def get_distance_matrix(adj_matrix):
    import networkx as nx

    leng = len(adj_matrix)
    G = nx.Graph(adj_matrix)
    lengths = dict(nx.all_pairs_shortest_path_length(G))  # 计算graph两两节点之间的最短路径的长度

    import numpy as np

    matrix = np.zeros([leng, leng], dtype=np.int)
    for i in lengths.keys():
        dist_dict = lengths[i]
        for j in dist_dict.keys():
            matrix[i][j] = dist_dict[j]
    return matrix


def get_dfg_edges(code, tree, flatten_ast, lang):
    """Get the data flow graph edges.

    Returns:
        dfg:
            [('a', 5, 'computedFrom', ['random'], [7]),
            ('a', 11, 'comesFrom', ['a'], [5]),
            ('b', 17, 'computedFrom', ['a', '1'], [19, 21]),
            ...]
    """
    if lang == "java":
        dfg_func = DFG_java

    # get dfg edges
    root_node = tree.root_node
    code = code.split("\n")
    index_to_code = get_index2code(code, flatten_ast)
    DFG, _ = dfg_func(root_node, index_to_code, {})
    DFG = sorted(DFG, key=lambda x: x[1])
    indexs = set()
    for d in DFG:
        if len(d[-1]) != 0:
            indexs.add(d[1])
        for x in d[-1]:
            indexs.add(x)
    new_DFG = []
    for d in DFG:
        if d[1] in indexs:
            new_DFG.append(d)
    return new_DFG


def get_df_matrix(dfg, flatten_ast, max_len=512):
    """
    Obtain the dataflow matrix.

    :param code:
    :param tree:
    :param flatten_ast:
    :param parser:
    :param lang:
    :return: dataflow matrix
    """
    length = len(flatten_ast)
    matrix = (
        np.zeros([length, length], dtype=np.bool)
        if length <= max_len
        else np.zeros([max_len, max_len], dtype=np.bool)
    )

    # get dfg matrix in the splited AST
    for edge in dfg:
        (tgt, tgt_idx, edge_type, src_list, src_idx_list) = edge
        for src_idx in src_idx_list:
            if src_idx < max_len and tgt_idx < max_len:
                matrix[src_idx][tgt_idx] = 1

    return matrix


def get_sequence(flatten_ast):
    """
    Get token or type sequences from the AST.

    :param flatten_ast:
    :return:
        tokens_seq: token sequence
        types_seq: type sequence
        non_leaf_idx: the index of non-leaf nodes
    """

    tokens_seq = []
    types_seq = []
    non_leaf_idx = []
    for idx, anyNode in enumerate(flatten_ast):
        if len(anyNode.children) > 0:
            non_leaf_idx.append(idx)
        token = anyNode.token if anyNode.token is not SPLITTED_WORD else anyNode.type
        typ = anyNode.type
        tokens_seq.append(token)
        types_seq.append(typ)
        assert (token != "") and (typ != "")

    return tokens_seq, types_seq, non_leaf_idx


def combine_multi_view_matrix(ast_matrix, sib_matrix, dfg_matrix):
    multi_matrix = ast_matrix + sib_matrix + dfg_matrix
    return multi_matrix


if __name__ == "__main__":
    print(split_identifier_into_parts("1MakeSence1_okk_SenseIt"))
