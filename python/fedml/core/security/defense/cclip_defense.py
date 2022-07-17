import random
from .defense_base import BaseDefenseMethod
from ..common import utils

"""
defense @ server, added by Xiaoyang, 07/10/2022
"Byzantine-Robust Learning on Heterogeneous Datasets via Bucketing"
https://arxiv.org/pdf/2006.09365.pdf
"""


class CClipDefense(BaseDefenseMethod):
    def __init__(self, tau, bucket_size=1):
        self.tau = tau  # clipping raduis
        # element # in each bucket; a grad_list is partitioned into floor(len(grad_list)/bucket_size) buckets
        self.bucket_size = bucket_size
        self.initial_guess = None

    def defend(self, client_grad_list, global_w=None):
        self.initial_guess = self._compute_an_initial_guess(client_grad_list)
        num_client = len(client_grad_list)
        vec_local_w = [
            (client_grad_list[i][0], utils.vectorize_weight(client_grad_list[i][1]))
            for i in range(0, num_client)
        ]
        vec_refs = utils.vectorize_weight(self.initial_guess)
        cclip_score = self._compute_cclip_score(vec_local_w, vec_refs)
        new_grad_list = []
        for i in range(num_client):
            tuple = dict()
            sample_num, local_params = client_grad_list[i]
            for k in local_params.keys():
                tuple[k] = (local_params[k] - self.initial_guess[k]) * cclip_score[i]
            new_grad_list.append((sample_num, tuple))
        print(f"cclip_score = {cclip_score}")
        return new_grad_list

    def robustify_global_model(self, avg_params, previous_global_w=None):
        for k in avg_params.keys():
            avg_params[k] = self.initial_guess[k] + avg_params[k]
        return avg_params

    @staticmethod
    def _compute_an_initial_guess(client_grad_list):
        # randomly select a gradient as the initial guess
        return client_grad_list[random.randint(0, len(client_grad_list))][1]

    def _compute_cclip_score(self, local_w, refs):
        cclip_score = []
        num_client = len(local_w)
        for i in range(0, num_client):
            dist = utils.compute_euclidean_distance(local_w[i][1], refs).item() + 1e-8
            score = min(1, self.tau / dist)
            cclip_score.append(score)
        return cclip_score