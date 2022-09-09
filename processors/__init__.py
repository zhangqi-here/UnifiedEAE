import json
from .processor_multiarg import MultiargProcessor


_DATASET_DIR = {
    'ace':{
        "train_file": './data/ace_eeqa/train.json',
        "dev_file": './data/ace_eeqa/dev.json',
        "test_file": './data/ace_eeqa/test.json',
        "max_span_num_file": "./data/dset_meta/role_num_ace.json",
    },
    'ace_wiki':{
        "train_file": './data/ace_eeqa/train.json#./data/WikiEvent/data_train_eeqa.json',
        "dev_file": './data/ace_eeqa/dev.json#./data/WikiEvent/data_dev_eeqa.json',
        "test_file": './data/ace_eeqa/test.json',
        "max_span_num_file": "./data/dset_meta/role_num_ace.json#./data/dset_meta/role_num_wikievent.json#./data/dset_meta/role_num_rams.json",
    },
    'ace_rams':{
        "train_file": './data/ace_eeqa/train.json#./data/RAMS_1.0/data/data_train_eeqa.json',
        "dev_file": './data/ace_eeqa/dev.json#./data/RAMS_1.0/data/data_dev_eeqa.json',
        "test_file": './data/WikiEvent/data_test_eeqa.json',
        "max_span_num_file": "./data/dset_meta/role_num_ace.json#./data/dset_meta/role_num_wikievent.json#./data/dset_meta/role_num_rams.json",
    },
    'wiki_rams':{
        "train_file": './data/WikiEvent/data_train_eeqa.json#./data/RAMS_1.0/data/data_train_eeqa.json',
        "dev_file": './data/RAMS_1.0/data/data_dev_eeqa.json',
        "test_file": './data/RAMS_1.0/data/data_test_eeqa.json',
        "max_span_num_file": "./data/dset_meta/role_num_ace.json#./data/dset_meta/role_num_wikievent.json#./data/dset_meta/role_num_rams.json",
    },
    'rams':{
        "train_file": './data/RAMS_1.0/data/data_train_eeqa.json',
        "dev_file": './data/RAMS_1.0/data/data_dev_eeqa.json',
        "test_file": './data/RAMS_1.0/data/data_test_eeqa.json',
        "max_span_num_file": "./data/dset_meta/role_num_rams.json",
    },
    "wikievent":{
        "train_file": './data/WikiEvent/data_train_eeqa.json',
        "dev_file": './data/WikiEvent/data_dev_eeqa.json',
        "test_file": './data/WikiEvent/data_test_eeqa.json',
        "max_span_num_file": "./data/dset_meta/role_num_wikievent.json",
    },
}


def build_processor(args, tokenizer):
    if args.dataset_type not in _DATASET_DIR: 
        raise NotImplementedError("Please use valid dataset name")
    args.train_file = _DATASET_DIR[args.dataset_type]['train_file']
    args.dev_file = _DATASET_DIR[args.dataset_type]['dev_file']
    args.test_file = _DATASET_DIR[args.dataset_type]['test_file']

    if args.model_type == "base":
        span_num_file = []
        span_file = "_DATASET_DIR[args.dataset_type]['max_span_num_file']".split("#")
        for file in span_file:
            with open(file) as f:
                span_num_file += json.load(f)
        args.max_span_num_dict = span_num_file

    processor = MultiargProcessor(args, tokenizer)
    return processor

