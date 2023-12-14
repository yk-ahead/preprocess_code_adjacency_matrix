import json
import re
import subprocess
import tokenize
from io import StringIO

from tqdm import tqdm


def get_parser(so, lang):
    from tree_sitter import Language, Parser

    LANGUAGE = Language(so, lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    return parser


def count_file_lines(file_path):
    """
    Counts the number of lines in a file using wc utility.
    :param file_path: path to file
    :return: int, no of lines
    """
    num = subprocess.check_output(["wc", "-l", file_path])
    num = num.decode("utf-8").split(" ")
    return int(num[0])


def read_source_files(filenames, num=None):
    with open(filenames["src_raw_train"]) as f:
        sources_raw_train = [
            line.strip() for line in tqdm(f, total=count_file_lines(filenames["src_raw_train"]))
        ]
    train_data = [json.loads(item)["raw_code"] for item in sources_raw_train]
    with open(filenames["src_raw_dev"]) as f:
        sources_raw_dev = [
            line.strip() for line in tqdm(f, total=count_file_lines(filenames["src_raw_dev"]))
        ]
    dev_data = [json.loads(item)["raw_code"] for item in sources_raw_dev]

    with open(filenames["src_raw_test"]) as f:
        sources_raw_test = [
            line.strip() for line in tqdm(f, total=count_file_lines(filenames["src_raw_test"]))
        ]
    test_data = [json.loads(item)["raw_code"] for item in sources_raw_test]

    if num is not None:
        dataset = {
            "train": train_data[:num],
            "valid": dev_data[:num],
            "test": test_data[:num],
        }
    else:
        dataset = {
            "train": train_data,
            "valid": dev_data,
            "test": test_data,
        }

    for key in dataset.keys():
        print("{} set: {}".format(key, len(dataset[key])))

    return dataset


def make_output_dirs(dirname):
    import os

    dist_npy_file_dir = os.path.join(dirname, "distances")
    mltv_npy_file_dir = os.path.join(dirname, "multiview")
    ast_npy_file_dir = os.path.join(dirname, "ast")
    df_npy_file_dir = os.path.join(dirname, "df")
    sib_npy_file_dir = os.path.join(dirname, "sib")

    _dirs_ = {
        "pre": dirname,
        "dist": dist_npy_file_dir,
        "mltv": mltv_npy_file_dir,
        "ast": ast_npy_file_dir,
        "df": df_npy_file_dir,
        "sib": sib_npy_file_dir,
    }
    for _dir_ in _dirs_.values():
        if not os.path.exists(_dir_):
            os.makedirs(_dir_)
    return _dirs_


def remove_comments_and_docstrings(source, lang):
    if lang in ["python"]:
        """
        Returns 'source' minus comments and docstrings.
        """
        io_obj = StringIO(source)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            ltext = tok[4]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += " " * (start_col - last_col)
            # Remove comments:
            if token_type == tokenize.COMMENT:
                pass
            # This series of conditionals removes docstrings:
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
                    # This is likely a docstring; double-check we're not inside an operator:
                    if prev_toktype != tokenize.NEWLINE:
                        if start_col > 0:
                            out += token_string
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        temp = []
        for x in out.split("\n"):
            if x.strip() != "":
                temp.append(x)
        return "\n".join(temp)
    elif lang in ["ruby"]:
        return source
    else:

        def replacer(match):
            s = match.group(0)
            if s.startswith("/"):
                return " "  # note: a space and not an empty string
            else:
                return s

        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"', re.DOTALL | re.MULTILINE
        )
        temp = []
        for x in re.sub(pattern, replacer, source).split("\n"):
            if x.strip() != "":
                temp.append(x)
        return "\n".join(temp)
