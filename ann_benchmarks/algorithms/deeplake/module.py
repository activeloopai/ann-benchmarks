import deeplake
from deeplake.core.vectorstore.vector_search.indra.index import METRIC_TO_INDEX_METRIC
from deeplake.core.vectorstore.deeplake_vectorstore import DeepLakeVectorStore
import os
import threading
import subprocess
import numpy as np
from ..base.module import BaseANN


# Class using the Deeplake implementation of an HNSW index for nearest neighbor
# search over data points in a high dimensional vector space.

class DeeplakeHnsw(BaseANN):
    def __init__(self, metric, param, enable_normalize = True, dimension = None):

        if metric not in ("angular", "euclidean"):
            raise NotImplementedError("Deeplake doesn't support metric %s" % metric)
        #self.metric = {"angular": METRIC_TO_INDEX_METRIC["COS"], "euclidean": METRIC_TO_INDEX_METRIC["L2"]}[metric]
        if metric == "angular":
            self.metric = "COS"
        else:
            self.metric = "L2"
        self.param = param
        self._ef_construction = param.get("efConstruction", 200)
        self._m = param.get("M", 16)
        self.dimension = dimension
        self.neighbors_to_explore = 200
        self.local_path = "ANN_benchmarks-embeddings"
        self.name = "Deeplake"
        self.my_token = os.environ.get('ACTIVELOOP_TOKEN')

    def list_open_files(self):
        pid = os.getpid()
        result = subprocess.check_output(['lsof', '-p', str(pid)])
        return result.decode('utf-8')


    def __del__(self):
        # Cleanup code here
        if hasattr(self, 'index'):
            del self.index

    def fit(self, X):
        self.index = DeepLakeVectorStore(path=self.local_path,
                                         overwrite=True,
                                         verbose=True,
                                         exec_option="compute_engine",
                                         index_params={"threshold": 5, "distance_metric": self.metric,
                                                       "additional_params":
                                                           {"efConstruction": self._ef_construction, "M": self._m, }},
                                         token=self.my_token)

        num_of_items, embedding_dim = X.shape
        ids = [f"{i}" for i in range(num_of_items)]
        # Creating embeddings of float32 as dtype of embedding tensor is float32.
        embedding = X
        text = ["aadfv" for i in range(num_of_items)]
        metadata = [{"key": i} for i in range(num_of_items)]

        self.index.add(
            id=ids,
            text=text,
            embedding=embedding,
            metadata=metadata,
        )
        ds2 = self.index.dataset
        emb_tensor = ds2.embedding
        self.index_al = emb_tensor.load_vdb_index("hnsw_1")

    def set_query_arguments(self, ef):
        return

    def query(self, v, n):
        v_float = np.array(v).astype(np.float32)
        view =  self.index_al.search_knn(v_float, n)
        return view.indices

    def __str__(self):
        return f"Deeplake(m={self._m}, ef_construction={self._ef_construction})"

    def freeIndex(self):
        print("Clearing out self.index")
        del self.index