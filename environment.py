import numpy as np
import math
# import matplotlib.pyplot as plt


from parameter import Parameters


class Env:
    def __init__(self, pa, nw_len_seqs=None, nw_size_seqs=None, nw_priority_seqs=None,
                 seed=42, render=False, repre='image', end='no_new_job'):
        #TODO: must change res_slot from an integer to an array of integers
        self.pa = pa
        self.render = render
        self.repre = repre  # image or compact representation
        self.end = end  # termination type, 'no_new_job' or 'all_done'

        self.nw_dist = pa.dist.bi_model_dist

        self.curr_time = 0

        # set up random seed
        if self.pa.unseen:
            np.random.seed(314159)
        else:
            np.random.seed(seed)

        if nw_len_seqs is None or nw_size_seqs is None:
            #MARK: changed
            # generate new work
            self.nw_len_seqs, self.nw_size_seqs, self.nw_priority_seqs = \
                self.generate_sequence_work(self.pa.simu_len * self.pa.num_ex)

            # self.workload = np.zeros(pa.num_res)
            # for i in range(pa.num_res):
            #     self.workload[i] = \
            #         np.sum(self.nw_size_seqs[:, i] * self.nw_len_seqs) / \
            #         float(pa.res_slot) / \
            #         float(len(self.nw_len_seqs))
            #     print("Load on # " + str(i) + " resource dimension is " + str(self.workload[i]))
            self.nw_len_seqs = np.reshape(self.nw_len_seqs,
                                          [self.pa.num_ex, self.pa.simu_len])
            #MARK: changed
            self.nw_size_seqs = np.reshape(self.nw_size_seqs,
                                           [self.pa.num_ex, self.pa.simu_len])
            self.nw_priority_seqs = np.reshape(self.nw_priority_seqs,
                                          [self.pa.num_ex, self.pa.simu_len])
        else:
            self.nw_len_seqs = nw_len_seqs
            self.nw_size_seqs = nw_size_seqs
            self.nw_priority_seqs = nw_priority_seqs

        self.seq_no = 0  # which example sequence
        self.seq_idx = 0  # index in that sequence
        biggest_job_in_sequence = np.amax(self.nw_size_seqs)
        # initialize system
        self.machine = Machine(pa,biggest_job_in_sequence)
        self.job_slot = JobSlot(pa)
        self.job_backlog = JobBacklog(pa)
        self.job_record = JobRecord()
        self.extra_info = ExtraInfo(pa)

    def generate_sequence_work(self, simu_len):
        nw_len_seq = np.zeros(simu_len, dtype=int)
        #Mark: changed
        nw_size_seq = np.zeros(simu_len, dtype=int)
        nw_priority_seq = np.zeros(simu_len, dtype=float)
        for i in range(simu_len):

            if np.random.rand() < self.pa.new_job_rate:  # a new job comes
                #TODO: change nw_dist
                nw_len_seq[i], nw_size_seq[i], nw_priority_seq[i] = self.nw_dist()
        return nw_len_seq, nw_size_seq, nw_priority_seq

    def get_new_job_from_seq(self, seq_no, seq_idx):
        #Mark: changed
        new_job = Job(res=self.nw_size_seqs[seq_no, seq_idx],
                      job_len=self.nw_len_seqs[seq_no, seq_idx],
                      job_id=len(self.job_record.record),
                      enter_time=self.curr_time,
                      priority=self.nw_priority_seqs[seq_no, seq_idx])
        return new_job

    def observe(self):
        if self.repre == 'image':

            backlog_width = int(math.ceil(self.pa.backlog_size / float(self.pa.time_horizon)))
            image_repr = np.zeros((self.pa.network_input_height, self.pa.network_input_width))

            ir_pt = 0

            for i in range(self.pa.num_res):

                image_repr[:, ir_pt: ir_pt + self.pa.res_slot] = self.machine.canvas[i, :, :]
                ir_pt += self.pa.res_slot

                for j in range(self.pa.num_nw):

                    if self.job_slot.slot[j] is not None:  # fill in a block of work
                        #TODO: change res_vec in job class
                        #Mark: changed
                        #MARK: priority added
                        image_repr[: self.job_slot.slot[j].len, ir_pt: ir_pt + self.job_slot.slot[j].res] = self.job_slot.slot[j].priority

                    ir_pt += self.pa.max_job_size
            #MARK: backlog changed (priority added)
            # image_repr[: int(self.job_backlog.curr_size / backlog_width), ir_pt: ir_pt + backlog_width] = 9
            # if self.job_backlog.curr_size % backlog_width > 0:
            #     image_repr[int(self.job_backlog.curr_size / backlog_width),
            #     ir_pt: ir_pt + self.job_backlog.curr_size % backlog_width] = 8

            backlog_index_to_fill = 0
            for job in self.job_backlog.backlog:
                if job is None:
                    continue
                image_repr[int(backlog_index_to_fill/backlog_width), ir_pt+int(backlog_index_to_fill%backlog_width)] = job.priority
                backlog_index_to_fill += 1

            ir_pt += backlog_width
            image_repr[:, ir_pt: ir_pt + 1] = self.extra_info.time_since_last_new_job / \
                                              float(self.extra_info.max_tracking_time_since_last_job)
            ir_pt += 1

            assert ir_pt == image_repr.shape[1]


            return image_repr

        # elif self.repre == 'compact':
        #
        #     compact_repr = np.zeros(self.pa.time_horizon * (self.pa.num_res + 1) +  # current work
        #                             self.pa.num_nw * (self.pa.num_res + 1) +  # new work
        #                             1,  # backlog indicator
        #                             dtype=theano.config.floatX)
        #
        #     cr_pt = 0
        #
        #     # current work reward, after each time step, how many jobs left in the machine
        #     job_allocated = np.ones(self.pa.time_horizon) * len(self.machine.running_job)
        #     for j in self.machine.running_job:
        #         job_allocated[j.finish_time - self.curr_time:] -= 1
        #
        #     compact_repr[cr_pt: cr_pt + self.pa.time_horizon] = job_allocated
        #     cr_pt += self.pa.time_horizon
        #
        #     # current work available slots
        #     for i in range(self.pa.num_res):
        #         compact_repr[cr_pt: cr_pt + self.pa.time_horizon] = self.machine.avbl_slot[:, i]
        #         cr_pt += self.pa.time_horizon
        #
        #     # new work duration and size
        #     for i in range(self.pa.num_nw):
        #
        #         if self.job_slot.slot[i] is None:
        #             compact_repr[cr_pt: cr_pt + self.pa.num_res + 1] = 0
        #             cr_pt += self.pa.num_res + 1
        #         else:
        #             compact_repr[cr_pt] = self.job_slot.slot[i].len
        #             cr_pt += 1
        #
        #             for j in range(self.pa.num_res):
        #                 compact_repr[cr_pt] = self.job_slot.slot[i].res_vec[j]
        #                 cr_pt += 1
        #
        #     # backlog queue
        #     compact_repr[cr_pt] = self.job_backlog.curr_size
        #     cr_pt += 1
        #
        #     assert cr_pt == len(compact_repr)  # fill up the compact representation vector
        #
        #     return compact_repr

    # def plot_state(self):
    #     plt.figure("screen", figsize=(20, 5))
    #
    #     skip_row = 0
    #
    #     for i in range(self.pa.num_res):
    #
    #         plt.subplot(self.pa.num_res,
    #                     1 + self.pa.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
    #                     i * (self.pa.num_nw + 1) + skip_row + 1)  # plot the backlog at the end, +1 to avoid 0
    #
    #         plt.imshow(self.machine.canvas[i, :, :], interpolation='nearest', vmax=1)
    #
    #         for j in range(self.pa.num_nw):
    #
    #             job_slot = np.zeros((self.pa.time_horizon, self.pa.max_job_size))
    #             if self.job_slot.slot[j] is not None:  # fill in a block of work
    #                 #MARK: changed
    #                 job_slot[: self.job_slot.slot[j].len, :self.job_slot.slot[j].res] = 1
    #
    #             plt.subplot(self.pa.num_res,
    #                         1 + self.pa.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
    #                         1 + i * (
    #                                     self.pa.num_nw + 1) + j + skip_row + 1)  # plot the backlog at the end, +1 to avoid 0
    #
    #             plt.imshow(job_slot, interpolation='nearest', vmax=1)
    #
    #             if j == self.pa.num_nw - 1:
    #                 skip_row += 1
    #
    #     skip_row -= 1
    #     backlog_width = int(math.ceil(self.pa.backlog_size / float(self.pa.time_horizon)))
    #     backlog = np.zeros((self.pa.time_horizon, backlog_width))
    #
    #     backlog[: self.job_backlog.curr_size / backlog_width, : backlog_width] = 1
    #     backlog[self.job_backlog.curr_size / backlog_width, : self.job_backlog.curr_size % backlog_width] = 1
    #
    #     plt.subplot(self.pa.num_res,
    #                 1 + self.pa.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
    #                 self.pa.num_nw + 1 + 1)
    #
    #     plt.imshow(backlog, interpolation='nearest', vmax=1)
    #
    #     plt.subplot(self.pa.num_res,
    #                 1 + self.pa.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
    #                 self.pa.num_res * (self.pa.num_nw + 1) + skip_row + 1)  # plot the backlog at the end, +1 to avoid 0
    #
    #     extra_info = np.ones((self.pa.time_horizon, 1)) * \
    #                  self.extra_info.time_since_last_new_job / \
    #                  float(self.extra_info.max_tracking_time_since_last_job)
    #
    #     plt.imshow(extra_info, interpolation='nearest', vmax=1)
    #
    #     plt.show()  # manual
    #     # plt.pause(0.01)  # automatic

    def get_reward(self):
        #MARK: changed (priority added)
        reward = 0
        for j in self.machine.running_job:
            reward += self.pa.delay_penalty / float(j.len)

        for j in self.job_slot.slot:
            if j is not None:
                priority = 1 - j.priority
                reward += (self.pa.hold_penalty / float(j.len)) / priority

        for j in self.job_backlog.backlog:
            if j is not None:
                priority = 1 - j.priority
                reward += (self.pa.dismiss_penalty / float(j.len)) / priority

        return reward

    def step(self, a, m, repeat=False):

        status = None

        done = False
        reward = 0
        info = None

        if a == self.pa.num_nw:  # explicit void action
            status = 'MoveOn'
        elif self.job_slot.slot[a] is None:  # implicit void action
            status = 'MoveOn'
        else:
            allocated = self.machine.allocate_job(self.job_slot.slot[a], m, self.curr_time)
            if not allocated:  # implicit void action
                status = 'MoveOn'
            else:
                status = 'Allocate'

        if status == 'MoveOn':
            self.curr_time += 1
            self.machine.time_proceed(self.curr_time)
            self.extra_info.time_proceed()

            # add new jobs
            self.seq_idx += 1

            if self.end == "no_new_job":  # end of new job sequence
                if self.seq_idx >= self.pa.simu_len:
                    done = True
            elif self.end == "all_done":  # everything has to be finished
                if self.seq_idx >= self.pa.simu_len and \
                        len(self.machine.running_job) == 0 and \
                        all(s is None for s in self.job_slot.slot) and \
                        all(s is None for s in self.job_backlog.backlog):
                    done = True
                elif self.curr_time > self.pa.episode_max_length:  # run too long, force termination
                    done = True

            if not done:

                if self.seq_idx < self.pa.simu_len:  # otherwise, end of new job sequence, i.e. no new jobs
                    new_job = self.get_new_job_from_seq(self.seq_no, self.seq_idx)

                    if new_job.len > 0:  # a new job comes

                        to_backlog = True
                        for i in range(self.pa.num_nw):
                            if self.job_slot.slot[i] is None:  # put in new visible job slots
                                self.job_slot.slot[i] = new_job
                                self.job_record.record[new_job.id] = new_job
                                to_backlog = False
                                break

                        if to_backlog:
                            if self.job_backlog.curr_size < self.pa.backlog_size:
                                self.job_backlog.backlog[self.job_backlog.curr_size] = new_job
                                self.job_backlog.curr_size += 1
                                self.job_record.record[new_job.id] = new_job
                            else:  # abort, backlog full
                                print("Backlog is full.")
                                # exit(1)

                        self.extra_info.new_job_comes()

            reward = self.get_reward()

        elif status == 'Allocate':
            self.job_record.record[self.job_slot.slot[a].id] = self.job_slot.slot[a]
            self.job_slot.slot[a] = None

            # dequeue backlog
            if self.job_backlog.curr_size > 0:
                self.job_slot.slot[a] = self.job_backlog.backlog[0]  # if backlog empty, it will be 0
                self.job_backlog.backlog[: -1] = self.job_backlog.backlog[1:]
                self.job_backlog.backlog[-1] = None
                self.job_backlog.curr_size -= 1

        ob = self.observe()

        info = self.job_record
        if done:
            self.seq_idx = 0

            if not repeat:
                self.seq_no = (self.seq_no + 1) % self.pa.num_ex

            self.reset()

        if self.render:
            self.plot_state()

        return ob, reward, done, info

    def reset(self):
        self.seq_idx = 0
        self.curr_time = 0

        # initialize system
        self.machine = Machine(self.pa,self.pa.res_slot)
        self.job_slot = JobSlot(self.pa)
        self.job_backlog = JobBacklog(self.pa)
        self.job_record = JobRecord()
        self.extra_info = ExtraInfo(self.pa)


class Job:
    def __init__(self, res, job_len, job_id, enter_time, priority):
        #MARK: changed
        self.id = job_id
        self.res = res
        self.len = job_len
        self.enter_time = enter_time
        self.start_time = -1  # not being allocated
        self.finish_time = -1
        self.priority = priority


class JobSlot:
    def __init__(self, pa):
        self.slot = [None] * pa.num_nw


class JobBacklog:
    def __init__(self, pa):
        self.backlog = [None] * pa.backlog_size
        self.curr_size = 0


class JobRecord:
    def __init__(self):
        self.record = {}


class Machine:
    def __init__(self, pa,biggest_job_in_sequence):
        self.num_res = pa.num_res
        self.time_horizon = pa.time_horizon
        self.res_slot = pa.res_slot
        # MARK: changed
        self.avbl_slot = np.ones((self.time_horizon, self.num_res))
        self.machine_sizes = np.ones(self.num_res)
        small_machine_chance = pa.dist.job_small_chance
        big_machine_upper = pa.res_slot
        big_machine_lower = pa.res_slot * 2 / 3
        small_machine_upper = pa.res_slot/2
        small_machine_lower = pa.res_slot/5
        # make sure there is at least one machine can do the biggest job
        s = np.random.randint(biggest_job_in_sequence,
                                 big_machine_upper + 1)
        self.machine_sizes[-1] *=  s
        for i in range(len(self.machine_sizes)-1):
            if np.random.rand() < small_machine_chance:
                size = np.random.randint(small_machine_lower,
                                         small_machine_upper + 1)
                self.machine_sizes[i] *= size
                self.avbl_slot[:, i] = self.avbl_slot[:, i] * size
            else:
                size = np.random.randint(big_machine_lower,
                                         big_machine_upper + 1)
                self.machine_sizes[i] *= size
                self.avbl_slot[:, i] = self.avbl_slot[:, i] * size
        with open('somefile.txt', 'a') as the_file:

            the_file.write("\nMachine sizes: %s\n" % self.machine_sizes)
        self.running_job = []

        # colormap for graphical representation
        self.colormap = np.arange(1 / float(pa.job_num_cap), 1, 1 / float(pa.job_num_cap))
        np.random.shuffle(self.colormap)

        # graphical representation
        self.canvas = np.zeros((pa.num_res, pa.time_horizon, pa.res_slot))

    def allocate_job(self, job, machine, curr_time):

        allocated = False

        for t in range(0, self.time_horizon - job.len):
            # MARK: changed (specific machine allocation)
            new_avbl_res = self.avbl_slot[t: t + job.len, machine] - job.res
            if np.all(new_avbl_res[:] >= 0):
                with open('somefile.txt', 'a') as the_file:
                    the_file.write("job size: %d  allocated to: %d  with size:%d\n---------------\n"%(job.res,machine,self.machine_sizes[machine]))
                allocated = True
                self.avbl_slot[t: t + job.len, machine] = new_avbl_res
                job.start_time = curr_time + t
                job.finish_time = job.start_time + job.len

                self.running_job.append(job)

                # update graphical representation

                used_color = np.unique(self.canvas[:])
                # WARNING: there should be enough colors in the color map
                for color in self.colormap:
                    if color not in used_color:
                        new_color = color
                        break

                assert job.start_time != -1
                assert job.finish_time != -1
                assert job.finish_time > job.start_time
                canvas_start_time = job.start_time - curr_time
                canvas_end_time = job.finish_time - curr_time
                #MARK: changed
                for i in range(canvas_start_time, canvas_end_time):
                    # get free indexes ( indexes in canvas for a resource in a specific time where is 0 )
                    avbl_slot = np.where(self.canvas[machine, i, :] == 0)[0]
                    # fill canvas for that resource in that specific time from first available index to as much as the resource needed for that job
                    self.canvas[machine, i, avbl_slot[: job.res]] = new_color

                break

        #MARK: changed
        # for r in range(0, self.num_res):
        #     for t in range(0, self.time_horizon - job.len):
        #         new_avbl_res = self.avbl_slot[t: t + job.len, r] - job.res
        #         if np.all(new_avbl_res[:] >= 0):
        #             allocated = True
        #             self.avbl_slot[t: t + job.len, r] = new_avbl_res
        #             job.start_time = curr_time + t
        #             job.finish_time = job.start_time + job.len
        #
        #             self.running_job.append(job)
        #
        #             # update graphical representation
        #
        #             used_color = np.unique(self.canvas[:])
        #             # WARNING: there should be enough colors in the color map
        #             for color in self.colormap:
        #                 if color not in used_color:
        #                     new_color = color
        #                     break
        #
        #             assert job.start_time != -1
        #             assert job.finish_time != -1
        #             assert job.finish_time > job.start_time
        #             canvas_start_time = job.start_time - curr_time
        #             canvas_end_time = job.finish_time - curr_time
        #             #MARK: changed
        #             for i in range(canvas_start_time, canvas_end_time):
        #                 # get free indexes ( indexes in canvas for a resource in a specific time where is 0 )
        #                 avbl_slot = np.where(self.canvas[r, i, :] == 0)[0]
        #                 # fill canvas for that resource in that specific time from first available index to as much as the resource needed for that job
        #                 self.canvas[r, i, avbl_slot[: job.res]] = new_color
        #
        #             break

        return allocated

    def time_proceed(self, curr_time):
        self.avbl_slot[:-1, :] = self.avbl_slot[1:, :]
        # MARK: changed
        self.avbl_slot[-1, :] = self.machine_sizes
        for job in self.running_job:

            if job.finish_time <= curr_time:
                self.running_job.remove(job)

        # update graphical representation

        self.canvas[:, :-1, :] = self.canvas[:, 1:, :]
        self.canvas[:, -1, :] = 0


class ExtraInfo:
    def __init__(self, pa):
        self.time_since_last_new_job = 0
        self.max_tracking_time_since_last_job = pa.max_track_since_new

    def new_job_comes(self):
        self.time_since_last_new_job = 0

    def time_proceed(self):
        if self.time_since_last_new_job < self.max_tracking_time_since_last_job:
            self.time_since_last_new_job += 1


# ==========================================================================
# ------------------------------- Unit Tests -------------------------------
# ==========================================================================


def test_backlog():
    pa = parameters.Parameters()
    pa.num_nw = 5
    pa.simu_len = 50
    pa.num_ex = 10
    pa.new_job_rate = 1
    pa.compute_dependent_parameters()

    env = Env(pa, render=False, repre='image')

    env.step(5)
    env.step(5)
    env.step(5)
    env.step(5)
    env.step(5)

    env.step(5)
    assert env.job_backlog.backlog[0] is not None
    assert env.job_backlog.backlog[1] is None
    print
    "New job is backlogged."

    env.step(5)
    env.step(5)
    env.step(5)
    env.step(5)

    job = env.job_backlog.backlog[0]
    env.step(0)
    assert env.job_slot.slot[0] == job

    job = env.job_backlog.backlog[0]
    env.step(0)
    assert env.job_slot.slot[0] == job

    job = env.job_backlog.backlog[0]
    env.step(1)
    assert env.job_slot.slot[1] == job

    job = env.job_backlog.backlog[0]
    env.step(1)
    assert env.job_slot.slot[1] == job

    env.step(5)

    job = env.job_backlog.backlog[0]
    env.step(3)
    assert env.job_slot.slot[3] == job

    print
    "- Backlog test passed -"


# def test_compact_speed():
#     pa = parameters.Parameters()
#     pa.simu_len = 50
#     pa.num_ex = 10
#     pa.new_job_rate = 0.3
#     pa.compute_dependent_parameters()
#
#     env = Env(pa, render=False, repre='compact')
#
#     import other_agents
#     import time
#
#     start_time = time.time()
#     for i in xrange(100000):
#         a = other_agents.get_sjf_action(env.machine, env.job_slot)
#         env.step(a)
#     end_time = time.time()
#     print
#     "- Elapsed time: ", end_time - start_time, "sec -"
#
#
# def test_image_speed():
#     pa = parameters.Parameters()
#     pa.simu_len = 50
#     pa.num_ex = 10
#     pa.new_job_rate = 0.3
#     pa.compute_dependent_parameters()
#
#     env = Env(pa, render=False, repre='image')
#
#     import other_agents
#     import time
#
#     start_time = time.time()
#     for i in xrange(100000):
#         a = other_agents.get_sjf_action(env.machine, env.job_slot)
#         env.step(a)
#     end_time = time.time()
#     print
#     "- Elapsed time: ", end_time - start_time, "sec -"


if __name__ == '__main__':
    # test_backlog()
    # test_compact_speed()
    # test_image_speed()
    pa = Parameters()
    pa.simu_len = 20
    pa.num_res = 2
    env = Env(pa)

    env.step(5,0)
    env.step(5,0)
    env.step(5,1)
    env.step(5,1)
    env.step(5,1)
    env.step(1,1)
    env.step(2,0)
    env.step(1,0)
    env.step(5,0)

    # env.step(0)
    # env.step(5)
    # env.step(5)
    # env.step(5)
    # env.step(5)
    # env.step(5)
    # env.step(5)
    # env.step(5)
    # env.step(5)
    # env.step(5)
    # env.step(5)
    # env.step(5)
    # env.step(5)
    # env.step(5)
    # env.step(5)
    # env.step(5)
    # env.step(5)
    # env.step(5)


    print(env.nw_len_seqs,"aalo",env.nw_size_seqs,env.nw_priority_seqs)
    print("********************")
    print(env.job_backlog.curr_size)
    np.savetxt("somefile.txt", env.observe(), fmt="%-5.2f")

