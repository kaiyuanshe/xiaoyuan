from __future__ import annotations
import json
import faiss

from datasets import load_dataset
from datasets.dataset_dict import DatasetDict
from datasets.arrow_dataset import Dataset
from datasets.search import NearestExamplesResults

from paddlenlp.transformers import AutoTokenizer, AutoModel
import os

from typing import List
import numpy as np
from tap import Tap
from numpy import ndarray
from paddlenlp.transformers import AutoTokenizer, BertModel, ErnieModel, ErnieTokenizer
import paddle
from paddle.static import InputSpec

import fastdeploy as fd
from loguru import logger

def read_json(file: str):
    import os
    if not os.path.exists(file):
        return None
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

class TextEncoderConfig(Tap):
    model_name: str = "rocketqa-zh-base-query-encoder"
    output_dir = "./output/"
    prefix = "rocketqa"
    vocab_path = ""
    device = "cpu"  # Type of inference device, support 'cpu' or 'gpu'.
    backend = 'onnx_runtime'    # 'onnx_runtime', 'paddle', 'openvino', 'tensorrt', 'paddle_tensorrt'
    batch_size = 1
    max_length = 64    # max sequence length
    log_interval = 10
    use_fp16 = False
    use_fast = False




def to_static(model, save_dir: str):
    inputs = [
        InputSpec(shape=[None, None])
    ]
    inputs = [
        paddle.static.InputSpec(shape=[None, None], dtype="int64"),
    ]

    model = paddle.jit.to_static(model, input_spec=inputs)
    # Save in static graph model.
    paddle.jit.save(model, save_dir)
        

class TextEncoder:
    def __init__(self, config: TextEncoderConfig):
        # init model
        file_path = os.path.join(config.output_dir, f"{config.prefix}.pdmodel")
        if not os.path.exists(file_path):
            model = AutoModel.from_pretrained(config.model_name)
            to_static(model, os.path.join(config.output_dir, config.prefix))

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, use_faster=config.use_fast)
        self.runtime = self.create_fd_runtime(config)
        self.batch_size = config.batch_size
        self.max_length = config.max_length

    def create_fd_runtime(self, config: TextEncoderConfig):
        option = fd.RuntimeOption()
        model_path = os.path.join(config.output_dir, f"{config.prefix}.pdmodel")
        params_path = os.path.join(config.output_dir, f"{config.prefix}.pdiparams")
        option.set_model_path(model_path, params_path)
        if config.device == 'cpu':
            option.use_cpu()
        else:
            option.use_gpu()
        if config.backend == 'paddle':
            option.use_paddle_infer_backend()
        elif config.backend == 'onnx_runtime':
            option.use_ort_backend()
        elif config.backend == 'openvino':
            option.use_openvino_backend()
        else:
            option.use_trt_backend()
            if config.backend == 'paddle_tensorrt':
                option.enable_paddle_to_trt()
                option.enable_paddle_trt_collect_shape()
            trt_file = os.path.join(config.output_dir, "infer.trt")
            option.set_trt_input_shape(
                'input_ids',
                min_shape=[1, config.max_length],
                opt_shape=[config.batch_size, config.max_length],
                max_shape=[config.batch_size, config.max_length])
            option.set_trt_input_shape(
                'token_type_ids',
                min_shape=[1, config.max_length],
                opt_shape=[config.batch_size, config.max_length],
                max_shape=[config.batch_size, config.max_length])
            if config.use_fp16:
                option.enable_trt_fp16()
                trt_file = trt_file + ".fp16"
            option.set_trt_cache_file(trt_file)
        return fd.Runtime(option)

    def preprocess(self, texts):
        data = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True)
        input_ids_name = self.runtime.get_input_info(0).name
        input_map = {
            input_ids_name: np.array(
                data["input_ids"], dtype="int64")
        }
        return input_map

    def encode(self, texts: List[str]) -> ndarray:
        input_map = self.preprocess(texts)
        result = self.runtime.infer(input_map)
        return result[1]

class KeywordRetriever():
    def __init__(self, corpus_file: str):
        self.corpus = read_json(corpus_file)

    def query(self, text: str):
        retrieved_examples = []
        for example in self.corpus:
            for keyword in example.get("keywords", "").split("/"):
                if not keyword:
                    continue

                sub_keywords = keyword.split("&")
                if all([sub_word in text for sub_word in sub_keywords]):
                    retrieved_examples.append(example)
                
        return retrieved_examples

class QAService:
    def __init__(self):
        corpus_file = './data/corpus-flat.json'
        self.config: TextEncoderConfig = TextEncoderConfig().parse_args(known_only=True)
        self.config.model_name = "rocketqa-zh-base-query-encoder"
        self.config.prefix = "rocketqa-zh-base-query-encoder"

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        self.dataset: DatasetDict = load_dataset("json", data_files=corpus_file)
        self.encoder = TextEncoder(self.config)

        self.feature_dataset: Dataset = self.dataset['train'].map(self.convert_features, batched=True, batch_size=100)
        self.feature_dataset.add_faiss_index("embedding", metric_type=faiss.METRIC_INNER_PRODUCT)

        self.threshold = 0.9

        self.answer_config: TextEncoderConfig = TextEncoderConfig().parse_args(known_only=True)
        self.answer_config.model_name = "rocketqa-base-cross-encoder"
        self.config.prefix = "rocketqa-base-cross-encoder"
        self.ranker: TextEncoder = TextEncoder(self.answer_config)
        
        self.keyword_retriever = KeywordRetriever(corpus_file)
    
    def convert_features(self, examples):
        embedding = self.encoder.encode(examples['question'])
        faiss.normalize_L2(embedding)

        return {
            "embedding": embedding
        }

    def query(self, text: str):
        # 1. retrivel from dpr
        embedding = self.encoder.encode([text])
        faiss.normalize_L2(embedding)
        result: NearestExamplesResults = self.feature_dataset.get_nearest_examples(
            "embedding",
            embedding,
            k=2
        )
        msg_result = []
        for index, score in enumerate(result.scores):
            if score > self.threshold:
                print(result.examples)
                msg_result.append(
                    f'{result.examples["answer"][index]}'
                )
        
        # 2. retrieve based on keyword
        examples = self.keyword_retriever.query(text)
        msg_result.extend([example["answer"] for example in examples])

        # 3. do rank on the result

        import os, psutil
        p = psutil.Process(int(os.getpid()))

        msg_info = "current memory usage: %dMB" % int(p.memory_info().rss / 1024 / 1024)
        logger.info(msg_info)
        return msg_result
