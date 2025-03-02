# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle
import time
from typing import List, Tuple

import faiss
import numpy as np
from tqdm import tqdm

class Indexer(object):

    def __init__(self, vector_sz, index_type='FlatIP', nlist=200000, n_subquantizers=64, n_bits=8):
        # vector_sz: embedding dimension
        self.index_type = index_type
        if index_type == 'IVFPQ':
            quantizer = faiss.IndexFlatIP(vector_sz)  # 使用内积作为量化器
            self.index = faiss.IndexIVFPQ(quantizer, vector_sz, nlist, n_subquantizers, n_bits)
        elif index_type == 'PQ':
            self.index = faiss.IndexPQ(vector_sz, n_subquantizers, n_bits, faiss.METRIC_INNER_PRODUCT)
        elif index_type == "FlatIP":
            self.index = faiss.IndexFlatIP(vector_sz)  # 默认使用IndexFlatIP
        else:
            raise NotImplementedError(f"unknown index_type: {index_type}")

        print(f'index_info:\nvector_sz:{vector_sz}\nindex_type:{index_type}\nnlist:{nlist}\nn_subquantizers:{n_subquantizers}\nn_bits:{n_bits}')
        
        #self.index_id_to_db_id = np.empty((0), dtype=np.int64)
        self.index_id_to_db_id = []

    def index_data(self, ids, embeddings):
        self._update_id_mapping(ids)
        # embeddings = embeddings.astype('float32')
        embeddings = np.array(embeddings, dtype=np.float32)
        if not self.index.is_trained:
            print(f'start training...')
            begin_time = time.time()
            self.index.train(embeddings)
            print(f'training time: {time.time() - begin_time}')
        self.index.add(embeddings)

        print(f'Total data indexed {len(self.index_id_to_db_id)}')

    def search_knn(self, query_vectors: np.array, top_docs: int, index_batch_size: int = 4096) -> List[Tuple[List[object], List[float]]]:
        # import pdb; pdb.set_trace()
        # query_vectors = query_vectors.astype('float32')
        query_vectors = np.array(query_vectors, dtype=np.float32)
        result = []
        nbatch = (len(query_vectors)-1) // index_batch_size + 1
        for k in tqdm(range(nbatch), desc=f"searching bz: {index_batch_size}"):
            start_idx = k*index_batch_size
            end_idx = min((k+1)*index_batch_size, len(query_vectors))
            q = query_vectors[start_idx: end_idx]
            # scores, indexes = self.index.search(n=q.shape[0], x=q, k=top_docs)
            scores, indexes = self.index.search(q, top_docs)
            # convert to external ids
            db_ids = [[str(self.index_id_to_db_id[i]) for i in query_top_idxs] for query_top_idxs in indexes]
            result.extend([(db_ids[i], scores[i]) for i in range(len(db_ids))])
        return result

    def serialize(self, dir_path):
        index_file = os.path.join(dir_path, f'{self.index_type}_index.faiss')
        meta_file = os.path.join(dir_path, f'{self.index_type}_index_meta.faiss')
        print(f'Serializing index to {index_file}, meta data to {meta_file}')

        faiss.write_index(self.index, index_file)
        with open(meta_file, mode='wb') as f:
            pickle.dump(self.index_id_to_db_id, f)

    def deserialize_from(self, dir_path):
        index_file = os.path.join(dir_path, f'{self.index_type}_index.faiss')
        meta_file = os.path.join(dir_path, f'{self.index_type}_index_meta.faiss')
        print(f'Loading index from {index_file}, meta data from {meta_file}')

        self.index = faiss.read_index(index_file)
        print('Loaded index of type %s and size %d', type(self.index), self.index.ntotal)

        with open(meta_file, "rb") as reader:
            self.index_id_to_db_id = pickle.load(reader)
        assert len(
            self.index_id_to_db_id) == self.index.ntotal, 'Deserialized index_id_to_db_id should match faiss index size'

    def _update_id_mapping(self, db_ids: List):
        #new_ids = np.array(db_ids, dtype=np.int64)
        #self.index_id_to_db_id = np.concatenate((self.index_id_to_db_id, new_ids), axis=0)
        self.index_id_to_db_id.extend(db_ids)