import os
import pathlib


def expand_braces(input: tuple[str]) -> tuple[str]:
    """
    Expand all braces of given strings and return tuple of all extended strings.
    For example, input ("a{b,c}d", "e{f,g}h") will return ("abd", "acd", "efh", "egh").

    Args:
        input (tuple[str]): Iterable of strings with braces to be expanded.

    Returns:
        tuple[str]: Tuple of expanded strings.
    """
    def brace_expand(s):
        # "a{b,c}d" -> ["abd", "acd"]
        if "{" not in s:
            return [s]
        pre, post = s.split("{", 1)
        mid, post = post.split("}", 1)
        parts = mid.split(",")
        expanded = []
        for part in parts:
            for rest in brace_expand(post):
                expanded.append(pre + part + rest)
        return expanded
    cols = []
    for _str in input:
        cols.extend(brace_expand(_str))
    return tuple(cols)


def prefix_map():
    stem = pathlib.Path(os.environ["INPUT_DATA_DIR"]).stem
    stem_to_prefix = {
    "prod14": "res_dnn_pnet",
    "prod20_vbf": "reg_dnn_moe",
    "prod20": "reg_dnn_moe",
    "prod19": "res_dnn_pnet",
    "prod24" : "reg_dnn_moe",
    "prod27_dyext": "reg_dnn_moe",
    }
    return stem_to_prefix[stem]

def expected_embedding_inputs():
    data_prefix = prefix_map()
    embedding_expected_inputs = {
        f"{data_prefix}_pair_type": [0, 1, 2],  # see mapping below
        f"{data_prefix}_dm1": [-1, 0, 1, 10, 11],  # -1 for e/mu
        f"{data_prefix}_dm2": [0, 1, 10, 11],
        f"{data_prefix}_vis_tau1_charge": [-1, 1],
        f"{data_prefix}_vis_tau2_charge": [-1, 1],
        f"{data_prefix}_has_fatjet": [0, 1],  # whether a selected fatjet is present
        f"{data_prefix}_has_jet_pair": [0, 1],  # whether two or more jets are present
        # 0: 2016APV, 1: 2016, 2: 2017, 3: 2018, 4: 2022preEE, 5: 2022postEE, 6: 2023pre, 7: 2023post
        f"{data_prefix}_year_flag": [0, 1, 2, 3, 4, 5, 6, 7],
        "channel_id": [1, 2, 3],
    }
    return embedding_expected_inputs


def add_prefix(string, prefix, ignore_code="_"):
    # small helper to add prefix to a string, but ignore it if it starts with ignore_code (e.g. "_")
    if string.startswith(ignore_code):
        return string[1:]
    else:
        return prefix + string

def feature(feature):
    data_prefix = prefix_map()
    feature =  [add_prefix(f, f"{data_prefix}_", ignore_code="_") for f in feature]
    return list(feature)

categorical_features = feature([
        "pair_type",
        # "_channel_id",
        "dm1",
        "dm2",
        "vis_tau1_charge",
        "vis_tau2_charge",
        "has_jet_pair",
        "has_fatjet",
])


continuous_features = feature([
    "met_px", "met_py",
    "met_cov00", "met_cov01", "met_cov11",
    "vis_tau1_px", "vis_tau1_py", "vis_tau1_pz", "vis_tau1_e",
    "vis_tau2_px", "vis_tau2_py", "vis_tau2_pz", "vis_tau2_e",
    "bjet1_px", "bjet1_py", "bjet1_pz", "bjet1_e",
    "bjet1_tag_b", "bjet1_tag_cvsb", "bjet1_tag_cvsl", "bjet1_hhbtag",
    "bjet2_px", "bjet2_py", "bjet2_pz", "bjet2_e",
    "bjet2_tag_b", "bjet2_tag_cvsb", "bjet2_tag_cvsl", "bjet2_hhbtag",
    "fatjet_px", "fatjet_py", "fatjet_pz", "fatjet_e",
    "htt_e", "htt_px", "htt_py", "htt_pz",
    "hbb_e", "hbb_px", "hbb_py", "hbb_pz",
    "htthbb_e", "htthbb_px", "htthbb_py", "htthbb_pz",
    "httfatjet_e", "httfatjet_px", "httfatjet_py", "httfatjet_pz",
    "nu1_px", "nu1_py", "nu1_pz",
    "nu2_px", "nu2_py", "nu2_pz",
])
