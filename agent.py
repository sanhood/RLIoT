import time
import threading
import numpy as np



from multiprocessing import Process
from multiprocessing import Manager

import environment
import job_distribution
import pg_network


def get_entropy(vec):
    entropy = - np.sum(vec * np.log(vec))
    if np.isnan(entropy):
        entropy = 0
    return entropy


def compute_discounted_R(Rewards, discount_rate=1):

    discounted_r = np.zeros_like(Rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(Rewards))):

        running_add = running_add * discount_rate + Rewards[t]

        discounted_r[t] = running_add

    discounted_r -= discounted_r.mean() / discounted_r.std()

    return discounted_r


def concatenate_all_ob(trajs, pa):

    timesteps_total = 0
    for i in range(len(trajs)):
        timesteps_total += len(trajs[i]['reward'])

    all_ob = np.zeros(
        (timesteps_total, pa.network_input_height * pa.network_input_width))

    timesteps = 0
    for i in range(len(trajs)):
        for j in range(len(trajs[i]['reward'])):
            all_ob[timesteps, :] = trajs[i]['ob'][j]
            timesteps += 1

    return all_ob


def concatenate_all_ob_across_examples(all_ob, pa):
    num_ex = len(all_ob)

    total_samp = 0
    for i in range(num_ex):
        total_samp += all_ob[i].shape[0]

    all_ob_contact = np.zeros(
        (total_samp, pa.network_input_height * pa.network_input_width))

    total_samp = 0

    for i in range(num_ex):
        prev_samp = total_samp

        total_samp += all_ob[i].shape[0]
        all_ob_contact[prev_samp : total_samp, :] = all_ob[i]

    return all_ob_contact


def get_traj(agent, env, episode_max_length):
    """
    Run agent-environment loop for one whole episode (trajectory)
    Return dictionary of results
    """

    env.reset()
    obs = []
    acts = []
    rews = []
    entropy = []
    info = []

    ob = env.observe()

    for _ in range(episode_max_length):

        act_prob = agent.get_action(ob)

        # csprob_n = np.cumsum(act_prob)
        # a = (csprob_n > np.random.rand()).argmax()
        # print(act_prob)
        ob = ob.reshape(ob.shape[0]*ob.shape[1])
        obs.append(ob)  # store the ob at current decision making step
        acts.append(act_prob)

        ob, rew, done, info = env.step(act_prob, repeat=True)

        rews.append(rew)

        entropy.append(get_entropy(act_prob))
        if done: break
    return {'reward': np.array(rews),
            'ob': np.array(obs),
            'action': np.array(acts),
            'entropy': entropy,
            'info': info
            }


def get_traj_worker(pg_learner, env, pa, result):

    trajs = []

    for i in range(pa.num_seq_per_batch):
        traj = get_traj(pg_learner, env, pa.episode_max_length)
        trajs.append(traj)

    all_ob = concatenate_all_ob(trajs, pa)

    # Compute discounted sums of rewards
    rets = [compute_discounted_R(traj["reward"], pa.discount) for traj in trajs]
    maxlen = max(len(ret) for ret in rets)
    padded_rets = [np.concatenate([ret, np.zeros(maxlen - len(ret))]) for ret in rets]

    # Compute time-dependent baseline
    baseline = np.mean(padded_rets, axis=0)

    # Compute advantage function
    advs = [ret - baseline[:len(ret)] for ret in rets]
    all_action = np.concatenate([traj["action"] for traj in trajs])
    all_adv = np.concatenate(advs)

    all_eprews = np.array([compute_discounted_R(traj["reward"], pa.discount)[0] for traj in trajs])  # episode total rewards

    all_eplens = np.array([len(traj["reward"]) for traj in trajs])  # episode lengths

    # All Job Stat
    enter_time, finish_time, job_len = process_all_info(trajs)
    finished_idx = (finish_time >= 0)
    all_slowdown = (finish_time[finished_idx] - enter_time[finished_idx]) / job_len[finished_idx]

    all_entropy = np.concatenate([traj["entropy"] for traj in trajs])

    result.append({"all_ob": all_ob,
                   "all_action": all_action,
                   "all_adv": all_adv,
                   "all_eprews": all_eprews,
                   "all_eplens": all_eplens,
                   "all_slowdown": all_slowdown,
                   "all_entropy": all_entropy})


def process_all_info(trajs):
    enter_time = []
    finish_time = []
    job_len = []

    for traj in trajs:
        enter_time.append(np.array([traj['info'].record[i].enter_time for i in range(len(traj['info'].record))]))
        finish_time.append(np.array([traj['info'].record[i].finish_time for i in range(len(traj['info'].record))]))
        job_len.append(np.array([traj['info'].record[i].len for i in range(len(traj['info'].record))]))

    enter_time = np.concatenate(enter_time)
    finish_time = np.concatenate(finish_time)
    job_len = np.concatenate(job_len)

    return enter_time, finish_time, job_len


def launch(pa, pg_resume=None, render=False, repre='image', end='no_new_job'):

    # ----------------------------
    print("Preparing for workers...")
    # ----------------------------

    pg_learners = []
    envs = []

    for ex in range(pa.num_ex):
        env = environment.Env(pa,
                              render=False, repre=repre, end=end)
        env.seq_no = ex
        envs.append(env)

    # for ex in range(pa.batch_size + 1):  # last worker for updating the parameters
    #
    #     print "-prepare for worker-", ex
    #
        pg_learner = pg_network.PGLearner(pa)
    #
    #     if pg_resume is not None:
    #         net_handle = open(pg_resume, 'rb')
    #         net_params = cPickle.load(net_handle)
    #         pg_learner.set_net_params(net_params)
    #
        pg_learners.append(pg_learner)

    # accums = init_accums(pg_learners[pa.batch_size])

    # --------------------------------------
    print("Preparing for reference data...")
    # --------------------------------------

    # ref_discount_rews, ref_slow_down = slow_down_cdf.launch(pa, pg_resume=None, render=False, plot=False, repre=repre, end=end)
    # mean_rew_lr_curve = []
    # max_rew_lr_curve = []
    # slow_down_lr_curve = []

    # --------------------------------------
    print("Start training...")
    # --------------------------------------

    timer_start = time.time()

    for iteration in range(1, pa.num_epochs):

        ps = []  # threads
        manager = Manager()  # managing return results
        manager_result = manager.list([])

        ex_indices = np.arange(pa.num_ex)
        np.random.shuffle(ex_indices)

        all_eprews = []
        grads_all = []
        loss_all = []
        eprews = []
        eplens = []
        all_slowdown = []
        all_entropy = []

        ex_counter = 0
        for ex in range(pa.num_ex):
            # print(ex)
            ex_idx = ex_indices[ex]
            # p = Process(target=get_traj,
            #             args=(pg_learners[ex_counter], envs[ex_idx], pa.episode_max_length, manager_result,))
            #
            # ps.append(p)
            # ex_counter += 1
            #
            # if ex_counter >= pa.batch_size or ex == pa.num_ex - 1:
            #
            #     ex_counter = 0
            #     for p in ps:
            #         p.start()
            #
            #     for p in ps:
            #         p.join()

            result = []  # convert list from shared memory



            ps = []
            result = []
            get_traj_worker(pg_learners[ex_counter], envs[ex_idx], pa, manager_result, )
            for r in manager_result:
                result.append(r)
            manager_result = manager.list([])
            # print("ok")
            all_ob = concatenate_all_ob_across_examples([r["all_ob"] for r in result], pa)
            all_action = np.concatenate([r["all_action"] for r in result])
            all_adv = np.concatenate([r["all_adv"] for r in result])
            pg_learners[0].fit(all_ob, all_action, all_adv)

            all_eprews.extend([r["all_eprews"] for r in result])
            # print(all_eprews)
            eprews.extend(np.concatenate([r["all_eprews"] for r in result]))  # episode total rewards
            eplens.extend(np.concatenate([r["all_eplens"] for r in result]))  # episode lengths
            all_slowdown.extend(np.concatenate([r["all_slowdown"] for r in result]))
            all_entropy.extend(np.concatenate([r["all_entropy"] for r in result]))
            #train the first agent






            timer_end = time.time()

            print ("-----------------")
            print ("Iteration: \t %i" % iteration)
            print ("NumTrajs: \t %i" % len(eprews))
            print ("NumTimesteps: \t %i" % np.sum(eplens))
            print ("Loss:     \t %s" % np.mean(loss_all))
            print ("MaxRew: \t %s" % np.average([np.max(rew) for rew in all_eprews]))
            print ("MeanRew: \t %s +- %s" % (np.mean(eprews), np.std(eprews)))
            print ("MeanSlowdown: \t %s" % np.mean(all_slowdown))
            print ("MeanLen: \t %s +- %s" % (np.mean(eplens), np.std(eplens)))
            print ("MeanEntropy \t %s" % (np.mean(all_entropy)))
            print ("Elapsed time\t %s" % (timer_end - timer_start), "seconds")
            print ("-----------------")



        #TODO: set paramaetes for other agents

        # for i in xrange(pa.batch_size + 1):
        #     pg_learners[i].set_net_params(params)

        timer_end = time.time()

        # print "-----------------"
        # print "Iteration: \t %i" % iteration
        # print "NumTrajs: \t %i" % len(eprews)
        # print "NumTimesteps: \t %i" % np.sum(eplens)
        # # print "Loss:     \t %s" % np.mean(loss_all)
        # print "MaxRew: \t %s" % np.average([np.max(rew) for rew in all_eprews])
        # print "MeanRew: \t %s +- %s" % (np.mean(eprews), np.std(eprews))
        # print "MeanSlowdown: \t %s" % np.mean(all_slowdown)
        # print "MeanLen: \t %s +- %s" % (np.mean(eplens), np.std(eplens))
        # print "MeanEntropy \t %s" % (np.mean(all_entropy))
        # print "Elapsed time\t %s" % (timer_end - timer_start), "seconds"
        # print "-----------------"
        #
        # timer_start = time.time()
        #
        # max_rew_lr_curve.append(np.average([np.max(rew) for rew in all_eprews]))
        # mean_rew_lr_curve.append(np.mean(eprews))
        # slow_down_lr_curve.append(np.mean(all_slowdown))
        #
        # if iteration % pa.output_freq == 0:
        #     param_file = open(pa.output_filename + '_' + str(iteration) + '.pkl', 'wb')
        #     cPickle.dump(pg_learners[pa.batch_size].get_params(), param_file, -1)
        #     param_file.close()
        #
        #     pa.unseen = True
        #     slow_down_cdf.launch(pa, pa.output_filename + '_' + str(iteration) + '.pkl',
        #                          render=False, plot=True, repre=repre, end=end)
        #     pa.unseen = False
        #     # test on unseen examples
        #
        #     plot_lr_curve(pa.output_filename,
        #                   max_rew_lr_curve, mean_rew_lr_curve, slow_down_lr_curve,
        #                   ref_discount_rews, ref_slow_down)


def main():

    import parameter

    pa = parameter.Parameters()

    pa.simu_len = 50  # 1000
    pa.num_ex = 1  # 100
    pa.num_nw = 10
    pa.num_seq_per_batch = 20
    pa.output_freq = 50
    pa.batch_size = 10

    # pa.max_nw_size = 5
    # pa.job_len = 5
    pa.new_job_rate = 0.3

    pa.episode_max_length = 2000  # 2000

    pa.compute_dependent_parameters()

    # pa.max_nw_size = 5
    # pa.job_len = 5



    pg_resume = None
    # pg_resume = 'data/tmp_450.pkl'

    render = False

    launch(pa, pg_resume, render, repre='image', end='all_done')


if __name__ == '__main__':
    main()
