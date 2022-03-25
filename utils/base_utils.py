import numpy.linalg as LA
import numpy as np
from sklearn.neighbors import NearestNeighbors
import time
from tqdm import tqdm
import faiss
import torch
import os
import sys
import math


EPS = 1e-7

def L2_normalize_numpy(x, verbose=False):
  """Row normalization
  Args:
    x: a numpy matrix of shape N*D
  Returns:
    x: L2 normalized x
  """
  sqr_row_sum = LA.norm(x, axis=1, keepdims=True)
  iszero = sqr_row_sum <= EPS
  if verbose:
    print(f'There are {iszero.sum()} zero-padding feature(s).')
  sqr_row_sum[iszero] = 1  # XJ: avoid division by zero
  y = x / sqr_row_sum
  del x, sqr_row_sum, iszero
  return y

def is_normalized(x):
  s = np.sum(x[0] * x[0])
  if s < 1 + EPS and s > 1 - EPS:
    return True
  else:
    return False


class silent_print():

    def __init__(self, suppress=True):
        self.suppress = suppress

    def __enter__(self):
        if self.suppress:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.suppress:
            sys.stdout.close()
            sys.stdout = self._original_stdout


def batch_search(index, query, batch_query_size=100000, topk=32, verbose=False):
    # pre allocate memory
    distance = np.zeros((len(query), topk), dtype=np.float32)
    neighbor = np.zeros((len(query), topk), dtype=np.int64)

    # search by batch
    for start_idx in tqdm(range(0, len(query), batch_query_size),
        desc="faiss searching...", disable=not(verbose)):
        end_idx = min(len(query), start_idx + batch_query_size)
        distance[start_idx:end_idx], neighbor[start_idx:end_idx] = index.search(query[start_idx:end_idx], topk)
    return distance, neighbor

class faiss_index_interface():
    def __init__(self, target, target_ids=None, index_factory_string=None, nprobe=32, num_gpu=None, verbose=False, mode='proxy', using_gpu=True):
        self._res_list = []
        print('In faiss_index_interface...')
        with silent_print(suppress=not verbose):
            # configure GPU resources
            gpu_avail = len(os.environ['CUDA_VISIBLE_DEVICES'].split(","))
            # print('gpu_avail')
            num_gpu = gpu_avail if num_gpu is None else num_gpu
            if gpu_avail == 0:
                raise RuntimeError("no GPU available, force terminated")
            if num_gpu > gpu_avail:
                print("gpu not enough, using {} GPU(s) to continue".format(gpu_avail))

            # setting faiss configuration
            size, dim = target.shape

            if size > 0:
                pq = 32 #if size < 40000000 else 16
                index_factory_string = "IVF{},PQ{}".format(min(8192, 16 * round(math.sqrt(size))), pq) if index_factory_string is None else index_factory_string
                cpu_index = faiss.index_factory(dim, index_factory_string)
                cpu_index.nprobe = nprobe

                # choose the right GPU index policy
                if mode == 'proxy':
                    co = faiss.GpuClonerOptions()
                    co.useFloat16 = True
                    co.usePrecomputed = False

                    index = faiss.IndexProxy()


                    for i in range(num_gpu):
                        if size > 0:
                            res = faiss.StandardGpuResources()
                            self._res_list.append(res)
                            sub_index = faiss.index_cpu_to_gpu(res, i, cpu_index, co) if using_gpu else cpu_index

                        index.addIndex(sub_index)

                elif mode == 'shard':
                    co = faiss.GpuMultipleClonerOptions()
                    co.useFloat16 = True
                    co.usePrecomputed = False
                    co.shard = True
                    index = faiss.index_cpu_to_all_gpus(cpu_index, co, ngpu=num_gpu)
                else:
                    raise ValueError("index mode unknown")

                index = faiss.IndexIDMap(index)
                index.verbose = verbose


                # get nlist to decide how many samples used for training
                nlist = int([item for item in index_factory_string.split(",") if 'IVF' in item][0].replace("IVF",""))

                # training
                if not index.is_trained:
                    t = time()
                    indexes_sample_for_train = np.random.randint(0, len(target), nlist * 256)
                    index.train(target[indexes_sample_for_train])
                    print(f'faiss trained, cost time {time() - t:.0f} s.')

                # add with ids
                print('Adding...')
                t = time()
                target_ids = np.arange(0, size) if target_ids is None else target_ids
                index.add_with_ids(target, target_ids)
                print(f'Adding time {time() - t:.0f} s.')
                self.index = index

            else:
                ngpu = num_gpu
                feat_dim = dim
                flat_config = []
                for i in range(ngpu):
                    cfg = faiss.GpuIndexFlatConfig()
                    cfg.useFloat16 = True #False
                    cfg.device = i
                    flat_config.append(cfg)

                res = [faiss.StandardGpuResources() for i in range(ngpu)]
                indexes = [faiss.GpuIndexFlatL2(res[i], feat_dim, flat_config[i]) for i in range(ngpu)]
                index = faiss.IndexProxy()

                for sub_index in indexes:
                    index.addIndex(sub_index)

                index.add(target)
                self.index = index



def faiss_search_neighbor(query, target=None, index_factory_string=None,
    topk=32, nprobe=32, num_gpu=None, batch_query_size=1000000, verbose=False):
    target = query if target is None else target
    # setup index on target
    index = faiss_index_interface(target, index_factory_string=index_factory_string, nprobe=nprobe, num_gpu=num_gpu, verbose=verbose)

    # search
    distance, neighbor = batch_search(index, query, topk=topk, verbose=verbose)
    # cleanup index on GPU
    del index
    return distance, neighbor

def fast_precise_similarity(query, neighbor, similarity, target=None,
                            process_unit=1000, sort=True, sort_unit=10000, verbose=False):
    if verbose:
        print('Precise calculating similarity...')
        start_time = time.time()
    query = torch.from_numpy(query)
    neighbor = torch.from_numpy(neighbor)
    similarity = torch.from_numpy(similarity)
    target = torch.from_numpy(target) if target is not None else query

    for s in tqdm(range(0, query.shape[0], process_unit), desc='batch matrix multiply', disable=not verbose):
        e = min(query.shape[0], s+process_unit)
        # much slower when explicitly writing "target_tmp = target[neighbor[s:e]].permute(0,2,1)"
        torch.bmm(query[s:e].unsqueeze(1), target[neighbor[s:e]].permute(0,2,1), out=similarity[s:e])

    if sort:
        for s in tqdm(range(0, similarity.shape[0], sort_unit), desc='similarity sorting...', disable=not verbose):
            e = min(similarity.shape[0], s+sort_unit)
            similarity[s:e], indices = torch.sort(similarity[s:e], dim=1, descending=True)
            neighbor[s:e] = torch.gather(neighbor[s:e], 1, indices)
    if verbose: print(f'Time of fast precise similarity: {time.time() - start_time} s')



def search_neighbor_sklearn(query, target=None, topk=32, algorithm='ball_tree'):
    if target is None:
        target = query
    topk = min(topk, target.shape[0])

    nbrs = NearestNeighbors(n_neighbors=topk, algorithm=algorithm).fit(target)
    similarity, neighbor = nbrs.kneighbors(query)
    similarity = (1 - similarity ** 2 / 2).astype(np.float32)
    return similarity, neighbor

def search_neighbor_faiss(query, target=None, topk=32, precise=True, sort=True, verbose=True):
    if verbose: start_time = time.time()
    # NOTE: when len(target) is small, faiss returns some invalid neighbors -1
    similarity, neighbor = faiss_search_neighbor(query=query, target=target, topk=topk, verbose=verbose)
    if verbose: print(f'Imprecise faiss search done! {time.time() - start_time} s')

    if not precise:
        similarity = 1 - similarity
    else:
        fast_precise_similarity(query, neighbor, similarity, target=target, sort=sort, verbose=verbose)
    return similarity, neighbor


def search_neighbor(query, target=None, topk=32, sort=True, algorithm=None, normed=False, verbose=True, **kwargs):
    """Search nearest neighbor
    Args:
      query: query features
      target: target feature, can be None
      topk: number of nearest neighbors
      sort: whether the nearest neighbors are sorted according to their similairty

    Return:
      similarity: matrix of cosine similarity with nearest neighbors
      neighbor:   the index matrix of nearest neighbors
    """
    if not algorithm: algorithm = 'faiss' if query.shape[0] > 10000 else 'ball_tree'
    if target is not None and target.shape[0] > 10000:
        algorithm = 'faiss'
    if not normed:
        query = L2_normalize_numpy(query, verbose=verbose)
        target = None if target is None else L2_normalize_numpy(target, verbose=verbose)

    if verbose: print(f'Using search algorithm: {algorithm}')
    if algorithm == 'ball_tree':
        # returns precise similarity
        similarity, neighbor = search_neighbor_sklearn(query, target, topk=topk, algorithm=algorithm)
    elif algorithm == 'faiss':
        similarity, neighbor = search_neighbor_faiss(query, target=target, topk=topk, sort=sort, verbose=verbose)
    else:
        raise NotImplementedError

    return similarity, neighbor


############### aggregation ################

def feature_aggregation(feat, aggr_topk=32, aggr_th=0.6, beta=0.5, verbose=False):
    """ Feature aggregation
    """
    assert beta < 1
    feat = L2_normalize_numpy(feat, verbose=verbose)
    similarity, neighbor = search_neighbor(query=feat, topk=aggr_topk, sort=False, normed=True, verbose=verbose)
    aggr_feat = neighbor_aggregation(feat, similarity, neighbor, beta=beta, aggr_topk=aggr_topk,
                                     aggr_th=aggr_th, verbose=verbose)
    del feat
    return aggr_feat


def neighbor_aggregation(feat, similarity, neighbor, beta=0.5, aggr_topk=32, aggr_th=0.6,
                         process_unit=1000, direct_average=True, verbose=False):
    """ Feature aggregation
    Aggregation function
    $$ X_i^{L+1}  =  \alpha  *  X^L_i  + (1 - \alpha) *  \sum_{j \in knn(i)}( w_{i,j} * X^L_j), $$
    where,
    $$ w_{i,j} = \frac{ cosine(X_i, X_j) }{ \sum_{ k \in knn(i) } {cosine(X_i, X_k)} }. $$

    Args:
    aggregation parameters.

    Returns:
    Aggregated feature
    """

    if verbose: start_time = time.time()

    # find affinity of topk, the first one is itself
    weight = similarity[:, 1:aggr_topk+1]
    neighbor = neighbor[:, 1:aggr_topk+1]
    if direct_average:
        weight = (weight >= aggr_th).astype(np.float32)
    else:
        weight = np.where(weight >= aggr_th, weight, 0.)

    # normalize weight
    weight_row_sum = np.sum(weight, axis=1, keepdims=True)
    weight_row_sum[weight_row_sum <= 0] = 1
    weight /= weight_row_sum

    weight = torch.from_numpy(weight)
    neighbor = torch.from_numpy(neighbor)
    feat = torch.from_numpy(feat)
    aggr_feat = torch.zeros_like(feat)

    # https://pytorch.org/docs/stable/torch.html#torch.baddbmm
    for s in tqdm(range(0, feat.shape[0], process_unit), desc='aggregating...', disable=not verbose):
        e = min(feat.shape[0], s+process_unit)
        torch.baddbmm(feat[s:e].unsqueeze(1), weight[s:e].unsqueeze(1), feat[neighbor[s:e]],
                      beta=1-beta, alpha=beta, out=aggr_feat[s:e])
    aggr_feat = aggr_feat.numpy()
    aggr_feat = L2_normalize_numpy(aggr_feat)

    if verbose: print(f'Time of neighbor aggregation: {time.time() - start_time} s')
    return aggr_feat


def GCN_aggregation(tmp_feat, num_layers=2, aggr_topk=32, aggr_th=0.6, beta=0.5, verbose=True):
    for layer in range(0, num_layers):
        tmp_feat = feature_aggregation(tmp_feat, aggr_topk=aggr_topk, aggr_th=aggr_th, beta=beta, verbose=verbose)
    return tmp_feat


######### build edges ############
def edge_generator(similarity, neighbor, threshold=0.6, MinPts=2):
    u, v = [], []
    for i in range(similarity.shape[0]):
        sim, nbr = similarity[i], neighbor[i]
        idx = (sim > threshold).nonzero()[0]
        if len(idx) < MinPts:
            continue
        for j in idx:
            if nbr[j] != i:
                u.append(i)
                v.append(nbr[j])
    return np.array([u, v]).T


