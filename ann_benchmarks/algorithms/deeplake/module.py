import deeplake
import numpy as np
import os
import random
import string
from ..base.module import BaseANN


# Class using the Deeplake implementation of an HNSW index for nearest neighbor
# search over data points in a high dimensional vector space.

class DeeplakeHnsw(BaseANN):
    def __init__(self, metric, param, enable_normalize = True, dimension = None):
        if metric not in ("angular", "euclidean"):
            raise NotImplementedError(f"Deeplake doesn't support metric {metric}")
        if metric == "angular":
            self.metric = "cosine_similarity"
        else:
            self.metric = "l2_norm"
        self.param = param
        self._ef_construction = param.get("efConstruction", 200)
        self._m = param.get("M", 16)
        self.dimension = dimension
        suffix = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8)) 
        self.local_path = f"ANN_benchmarks-embeddings_{suffix}"
        self.name = "deeplake"
        self.token = os.environ.get('ACTIVELOOP_TOKEN')

    def __del__(self):
        self.freeIndex()

    def fit(self, X):
        self.ds = deeplake.dataset(self.local_path, overwrite=True, token=self.token)
        self.ds.create_tensor("embedding", htype="embedding", dtype="float32")
        self.ds.embedding.extend(X)
        self.ds.embedding.create_vdb_index("hnsw_1", distance=self.metric, additional_params={
            "efConstruction": self._ef_construction, "M": self._m
        })
        self.index = self.ds.embedding.load_vdb_index("hnsw_1")

    def set_query_arguments(self, ef):
        self.index.set_search_params(ef=ef)

    def query(self, v, n):
        v_float = np.array(v).astype(np.float32)
        view =  self.index.search_knn(v_float, n)
        return view.indices

    def __str__(self):
        return f"Deeplake(m={self._m}, ef_construction={self._ef_construction})"

    def freeIndex(self):
        if hasattr(self, 'index'):
            del self.index
        if hasattr(self, 'ds'):
            del self.ds
            deeplake.delete(self.local_path)
        if os.path.isfile(f"/tmp/{self.local_path}/embedding"):
            os.remove(f"/tmp/{self.local_path}/embedding")
