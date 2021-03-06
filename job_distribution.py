import numpy as np
import random

class Dist:

    def __init__(self, num_res, max_nw_size, job_len, num_job_priorities):
        self.num_res = num_res
        self.max_nw_size = max_nw_size
        self.job_len = job_len

        self.job_small_chance = 0.8

        self.job_len_big_lower = job_len * 2 / 3
        self.job_len_big_upper = job_len

        self.job_len_small_lower = 1
        self.job_len_small_upper = job_len / 5

        self.dominant_res_lower = max_nw_size / 2
        self.dominant_res_upper = max_nw_size

        self.other_res_lower = 1
        self.other_res_upper = max_nw_size / 5

        self.max_priority = num_job_priorities * 0.1

    def normal_dist(self):
        print('normal')
        # new work duration
        nw_len = np.random.randint(1, self.job_len + 1)  # same length in every dimension

        nw_size = np.zeros(self.num_res)

        for i in range(self.num_res):
            nw_size[i] = np.random.randint(1, self.max_nw_size + 1)

        return nw_len, nw_size

    def bi_model_dist(self):
        # -- job length --
        #MARK: changed
        nw_size = 0
        nw_len = 0
        priority = float('%.1f'%(np.random.uniform(0.1, self.max_priority)))
        if np.random.rand() < self.job_small_chance:  # small job
            nw_len = np.random.randint(self.job_len_small_lower,
                                       self.job_len_small_upper + 1)
            nw_size = np.random.randint(self.other_res_lower,
                                        self.other_res_upper + 1)
        else:  # big job
            nw_len = np.random.randint(self.job_len_big_lower,
                                       self.job_len_big_upper + 1)
            nw_size = np.random.randint(self.dominant_res_lower,
                                        self.dominant_res_upper + 1)
        return nw_len, nw_size, priority

#MARK: changed
# def generate_sequence_work(pa, seed=42):
#
#     np.random.seed(seed)
#
#     simu_len = pa.simu_len * pa.num_ex
#
#     nw_dist = pa.dist.bi_model_dist
#
#     nw_len_seq = np.zeros(simu_len, dtype=int)
#     nw_size_seq = np.zeros(simu_len, dtype=int)
#
#     for i in range(simu_len):
#
#         if np.random.rand() < pa.new_job_rate:  # a new job comes
#
#             nw_len_seq[i], nw_size_seq[i, :] = nw_dist()
#
#     nw_len_seq = np.reshape(nw_len_seq,
#                             [pa.num_ex, pa.simu_len])
#     nw_size_seq = np.reshape(nw_size_seq,
#                              [pa.num_ex, pa.simu_len, pa.num_res])
#
#     return nw_len_seq, nw_size_seq