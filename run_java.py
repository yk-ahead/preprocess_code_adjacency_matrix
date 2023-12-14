import multiprocessing
import os

import joblib
import myast
import numpy as np
import utils
from tqdm import tqdm


def generate_multi_view_matrix_java(flatten_ast, dfg):
    # get dataflow matrix
    dfg_matrix = myast.get_df_matrix(dfg, flatten_ast)

    # get ast parent-child relation matrix
    ast_matrix = myast.get_ast_matrix(flatten_ast)

    # get distance matrix
    distance_matrix = myast.get_distance_matrix(ast_matrix)

    # get ast siblings relation matrix
    sib_matrix = myast.get_sib_matrix(flatten_ast)

    # get ast tokens/types sequence
    token_seq, type_seq, non_leaf_idx = myast.get_sequence(flatten_ast)

    # merge matrix
    mul_view_matrix = myast.combine_multi_view_matrix(ast_matrix, sib_matrix, dfg_matrix)

    res = {
        "ast_matrix": ast_matrix,
        "dataflow_matrix": dfg_matrix,
        "siblings_matrix": sib_matrix,
        "mul_view_matrix": mul_view_matrix,
        "distance_matrix": distance_matrix,
        "token_seq": token_seq,
        "type_seq": type_seq,
        "non_leaf_idx": non_leaf_idx,
    }
    return res


def _pre_process(code, tree):
    flatten_ast = myast.get_flatten_ast(tree, split=True)
    dfg = myast.get_dfg_edges(code, tree, flatten_ast, lang="java")
    return flatten_ast, dfg


def _make_matrix(flatten_ast, dfg, out_dirs, uid):
    res = generate_multi_view_matrix_java(flatten_ast, dfg)
    np.savez_compressed(os.path.join(out_dirs["mltv"], f"{uid}.npy"), res["mul_view_matrix"])
    np.savez_compressed(os.path.join(out_dirs["ast"], f"{uid}.npy"), res["ast_matrix"])
    np.savez_compressed(os.path.join(out_dirs["df"], f"{uid}.npy"), res["dataflow_matrix"])
    np.savez_compressed(os.path.join(out_dirs["sib"], f"{uid}.npy"), res["siblings_matrix"])
    np.savez_compressed(os.path.join(out_dirs["dist"], f"{uid}.npy"), res["distance_matrix"])
    return (
        " ".join(res["token_seq"]) + "\n",
        " ".join(res["type_seq"]) + "\n",
        " ".join([str(id) for id in res["non_leaf_idx"]]) + "\n",
        str(uid) + "\n",
    )


def main():
    lang = "java"
    n_cpu = multiprocessing.cpu_count()

    # set parser
    parser = utils.get_parser("build/yk-languages.so", lang)

    # read the source files
    source_dir = "JAVA/tlcodesum/"
    filenames = {
        "src_raw_train": os.path.join(source_dir, "train/raw_code.train"),
        "src_raw_dev": os.path.join(source_dir, "valid/raw_code.valid"),
        "src_raw_test": os.path.join(source_dir, "test/raw_code.test"),
    }
    dataset = utils.read_source_files(filenames)

    # make the output directories
    pre_dir = os.path.join("processed_output", "tlcodesum", lang)
    out_dirs = utils.make_output_dirs(pre_dir)

    already = 0
    for key in dataset:
        n_samples = len(dataset[key])

        # pre-parse the ast for acceleration,
        # use backend="threading" as tree-sitter.node object cannot be pickled
        parallel_1 = joblib.Parallel(n_jobs=n_cpu, backend="threading")
        func_1 = joblib.delayed(_pre_process)
        res_1 = parallel_1(
            func_1(code, parser.parse(bytes(code, "utf8")))
            for code in tqdm(
                dataset[key],
                desc=f"{key}[1/2] {n_samples}  cases",
            )
        )
        flatten_asts, DFGs = zip(*res_1)

        # save matrix, use default backend="loky" which is much faster than "threading"
        parallel_2 = joblib.Parallel(n_jobs=n_cpu)
        func_2 = joblib.delayed(_make_matrix)
        res = parallel_2(
            func_2(flatten_asts[idx], DFGs[idx], out_dirs, idx + already)
            for idx in tqdm(
                range(n_samples),
                total=n_samples,
                desc=f"{key}[2/2] {n_samples}  cases",
            )
        )

        # save the token/type sequence
        code_f = open(os.path.join(out_dirs["pre"], f"{key}.token.code"), "w", encoding="utf8")
        nlid_f = open(os.path.join(out_dirs["pre"], f"{key}.idx.non_leaf"), "w", encoding="utf8")
        guid_f = open(os.path.join(out_dirs["pre"], f"{key}.idx.matrix"), "w", encoding="utf8")
        token_seq_list, type_seq_list, non_leaf_idx_list, guid = zip(*res)
        code_f.writelines(token_seq_list)
        nlid_f.writelines(non_leaf_idx_list)
        guid_f.writelines(guid)

        code_f.close()
        nlid_f.close()
        guid_f.close()
        already += n_samples


def test():
    lang = "java"
    parser = utils.get_parser("build/yk-languages.so", lang)
    code = """private int currentDepth ( ) { try { Integer oneBased = ( ( Integer ) DEPTH_FIELD . get ( this ) ) ; return oneBased - _NUM ; } catch ( IllegalAccessException e ) { throw new AssertionError ( e ) ; } }"""
    tree = parser.parse(bytes(code, "utf8"))
    flatten_ast = myast.get_flatten_ast(tree, split=True)
    dfg = myast.get_dfg_edges(code, tree, flatten_ast, lang="java")
    res = generate_multi_view_matrix_java(flatten_ast, dfg)
    print("OK")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # test()
    main()
