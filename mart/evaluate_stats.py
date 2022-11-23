"""
Get statistics (vocabulary size, average length etc.). Required.

References:
    Copyright (c) 2017 Ranjay Krishna
    Licensed under The MIT License, see https://choosealicense.com/licenses/mit/
    @inproceedings{krishna2017dense,
        title={Dense-Captioning Events in Videos},
        author={Krishna, Ranjay and Hata, Kenji and Ren, Frederic and Fei-Fei, Li and Niebles, Juan Carlos},
        booktitle={ArXiv},
        year={2017}
    }

    History:
    https://github.com/ranjaykrishna/densevid_eval
    https://github.com/jamespark3922/densevid_eval
    https://github.com/jayleicn/recurrent-transformer
    Current version 2021 https://github.com/gingsi/coot-videotext
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import nltk


def get_sen_stat(list_of_str):
    """
    list_of_str, list(str), str could be a sentence a paragraph
    """
    tokenized = [nltk.tokenize.word_tokenize(sen.lower()) for sen in list_of_str]
    num_sen = len(list_of_str)
    lengths = [len(e) for e in tokenized]
    # TODO : ここ必要か確認しておく
    if len(lengths) != 0:
        avg_len = 1.0 * sum(lengths) / len(lengths)
    else:
        avg_len = 1.0 * sum(lengths)
    full_vocab = set([item for sublist in tokenized for item in sublist])
    return {"vocab_size": len(full_vocab), "avg_sen_len": avg_len, "num_sen": num_sen}

# TODO : bilas / bila
def evaluate_stats_files(
    submission_file: str,
    reference_file: str,
    output_file: Optional[Union[str, Path]] = None,
    verbose: bool = False,
    datatype = "bilas",
) -> Dict[str, Any]:
    """
    Get vocab size, average length, etc
    
    bila
        submission_file: experiments/.../caption/translations_0_test.json
        ref_file: annotations/BILA/captioning_test_para.json
    bilas
        submisiion_file: experiments/...caption/translations_0_val.json
        ref_file : data/BilaS/caption_valid.json
    """
    # load data
    sub_data = json.load(open(submission_file, "r"))
    ref_data = json.load(open(reference_file, "r"))
    # print('------------ sub-data ---------')
    # print(sub_data)
    # print('--------- ref data ---------------')
    # print(ref_data)
    # print('--------- bf ------------------')
    sub_data = sub_data["results"] if "results" in sub_data else sub_data
    ref_data = ref_data["results"] if "results" in ref_data else ref_data
    # print('##### sub_data.items() ######')
    # print(sub_data.items())
    # print('-------------------------------------')
    # print(list(sub_data.items())[0])
    # print('----------------- ##### --------------')
    if datatype == "bila":
        sub_data = {k: v for k, v in list(sub_data.items()) if k in ref_data}
    elif datatype == 'bilas':
        sub_data = {k: v for k, v in list(sub_data.items()) if str(int(k)) in ref_data}

    # print('------- sub_data --------')
    # print(sub_data)
    # print('--------------------------')

    submission_data_entries = [
        item for sublist in list(sub_data.values()) for item in sublist
    ]
    # print(submission_data_entries)

    submission_sentences = [e["sentence"] for e in submission_data_entries]
    # print('------- submission data entries-------')
    # print(submission_data_entries)
    # print('------- submission sentences -------')
    # print(submission_sentences)
    # print('---------------------------------')
    submission_stat = get_sen_stat(submission_sentences)

    if verbose:
        for k in submission_stat:
            print(("{} submission {}".format(k, submission_stat[k])))
    final_res = {"submission": submission_stat}

    if "gt_sentence" in submission_data_entries[0]:
        gt_sentences = [e["gt_sentence"][0] for e in submission_data_entries]
        gt_stat = get_sen_stat(gt_sentences)  # only one reference is used here!!!
        final_res["gt_stat"] = gt_stat

    if output_file is not None:
        with open(output_file, "w", encoding="utf8") as f:
            if datatype == "bila":
                f.write(json.dumps(final_res, indent=4, sort_keys=True))
            elif datatype == "bilas":
                f.write(json.dumps(final_res, indent=4, sort_keys=True, ensure_ascii=False))
    
    return final_res
