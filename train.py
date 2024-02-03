import json

import hydra
import os
import torch
import replay_buffer
import numpy as np
from copy import deepcopy
from experiment_logging import default_logger as logger
from tqdm import tqdm
import random

@hydra.main(config_path='config', config_name='train')
def train(cfg):
    print('jobname: ', cfg.name)

    # load data
    # replay = torch.load(cfg.data_path)
    test_bed_path = cfg.data_path + '/' + 'testbed_data'

    # load env
    import gym
    # import d4rl

    cfg.state_dim = int(np.prod((150,)))
    cfg.action_dim = int(np.prod((1,)))

    # set seed = 0
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # build learners
    q = hydra.utils.instantiate(cfg.q)
    pi = hydra.utils.instantiate(cfg.pi)
    beta = hydra.utils.instantiate(cfg.beta)
    baseline = hydra.utils.instantiate(cfg.baseline)

    # print(type(beta))

    # setup logger 
    os.makedirs(cfg.log_dir, exist_ok=True)
    setup_logger(cfg)
    q.set_logger(logger)
    pi.set_logger(logger)
    beta.set_logger(logger)
    baseline.set_logger(logger)

    print ('beta step is:',cfg.beta_steps)
    print ('q step is:',cfg.q_steps)
    print ('pi step is:',cfg.pi_steps)

    # train
    if cfg.pi.name == 'pi_easy_bcq':
        pi.update_beta(beta)
        pi.update_q(q)


    beta_loss = []
    # train beta
    if cfg.train_beta:
        # train_beta_iterator = tqdm(range(int(cfg.beta_steps)),desc='training beta')
        # print (train_beta_iterator)
        for step in tqdm(range(int(cfg.beta_steps))):
            # print (step)
            beta.train_step(cfg.data_path, None, None, None)
            beta_loss.append(beta)

            if step % int(cfg.log_freq) == 0:
                logger.update('beta/step', step)
                # beta.eval(env, cfg.eval_episodes)
                # logger.write_sub_meter('beta')
            if step % int(cfg.beta_save_freq) == 0 and step != 0:
                beta.save(cfg.beta.model_save_path + '_' + str(step) + '.pt')
                print ('beta model is saved')
    print ('beta train is complete')
    beta_loss = beta.get_loss()
    # with open('./submission/test4' + '/beta_loss.json','w')as f:
    #     json.dump(beta_loss, f)

    # train baseline
    if cfg.train_baseline:
        for step in range(int(cfg.baseline_steps)):
            baseline.train_step(cfg.data_path)

            if step % int(cfg.log_freq) == 0:
                logger.update('baseline/step', step)
                # baseline.eval(env, beta, cfg.eval_episodes)
                # logger.write_sub_meter('baseline')
            if step % int(cfg.beta_save_freq) == 0:
                beta.save(cfg.beta.model_save_path + '_' + str(step) + '.pt')
    if cfg.train_beta:
        q.save(cfg.beta.model_save_path + '.pt')
    # load beta as init pi
    pi.load_from_pilearner(beta)


    for out_step in range(int(cfg.steps)):
        # train Q
        if cfg.train_q:
            q_loss = []
            for in_step in tqdm(range(int(cfg.q_steps))):
                q.train_step(cfg.data_path, pi, beta)
                q_loss.append(q)
                step = out_step * int(cfg.q_steps) + in_step 
                if step % int(cfg.log_freq) == 0:
                    logger.update('q/step', step)
                    # q.eval(env, pi, cfg.eval_episodes)
                    # logger.write_sub_meter('q')
                
                if step % int(cfg.q_save_freq) == 0:
                    q.save(cfg.q.model_save_path + '_' + str(step) + '.pt')
                    print ('q model is saved')

        print ('q training is complete')
        q_loss = q.get_loss()
        # with open('./submission/test4' + '/q_loss.json', 'w') as f:
        #     json.dump(q_loss, f)
        # train pi
        if cfg.train_pi and cfg.pi.name != 'pi_easy_bcq':

            for in_step in tqdm(range(int(cfg.pi_steps))):
                pi.train_step(cfg.data_path, q, baseline, beta)

                step = out_step * int(cfg.pi_steps) + in_step
                if step % int(cfg.log_freq) == 0:
                    logger.update('pi/step', step)
                    # pi.eval(env, cfg.eval_episodes)
                    # logger.write_sub_meter('pi')
                if step % int(cfg.pi_save_freq) == 0:
                    pi.save(cfg.pi.model_save_path + '_' + str(step) + '.pt')
                    print ('pi model is saved')
        elif cfg.pi.name == 'pi_easy_bcq':
            step = out_step + 1
            pi.update_q(q)
            if step % int(cfg.log_freq) == 0:
                logger.update('pi/step', step)
                # pi.eval(env, cfg.eval_episodes)
                # logger.write_sub_meter('pi')
    print ('pi training is complete')
    pi_loss = pi.get_pi_loss()
    # with open('./submission/test4' + '/pi_loss.json', 'w') as f:
    #     json.dump(pi_loss, f)

    if cfg.train_q:
        q.save(cfg.q.model_save_path + '.pt')
    if cfg.train_pi:
        pi.save(cfg.pi.model_save_path + '.pt')

     
def setup_logger(cfg):
    logger_dict = dict()
    if cfg.train_q:
        q_train_dict = {'q': {
                        'csv_path': f'{cfg.log_dir}/q.csv',
                        'format_str': cfg.q.format_str,
                    },} 
        logger_dict.update(q_train_dict)
    if cfg.train_pi or cfg.pi.name == 'pi_easy_bcq':
        pi_train_dict = {'pi': {
                        'csv_path': f'{cfg.log_dir}/pi.csv',
                        'format_str': cfg.pi.format_str,
                    },} 
        logger_dict.update(pi_train_dict)
    if cfg.train_beta:
        beta_train_dict = {'beta': {
                        'csv_path': f'{cfg.log_dir}/beta.csv',
                        'format_str': cfg.beta.format_str,
                    },} 
        logger_dict.update(beta_train_dict)
    if cfg.train_baseline:
        beta_train_dict = {'baseline': {
                        'csv_path': f'{cfg.log_dir}/baseline.csv',
                        'format_str': cfg.baseline.format_str,
                    },} 
        logger_dict.update(beta_train_dict)

    logger.setup(logger_dict, summary_format_str=None) 


if __name__ == "__main__":
    train()