import argparse
import time
from client import Client
import sys
from algorithms import (ac,
                        replay_memory,
                        stallion)
import numpy as np
import torch
from torch.autograd import Variable
import logging

sys.path.append("..")
import os
from utils.config import Config
from utils.utils import *

parser = argparse.ArgumentParser(description='Streaming client')
parser.add_argument('--ip', default='127.0.0.1', type=str, help='ip')
parser.add_argument('--port', default='8080', type=str, help='port')
parser.add_argument('--algo', default='stallion', type=str, help='ABR algorithm')
parser.add_argument('--sleep', default=0, type=float, help='Wait time')
args = parser.parse_args()

S_INFO = 7  # latency, idle, buffer_size, freeze, download_time, bw, last_rate
S_LEN = 8  # take how many gops in the past
A_DIM = len(Config.BITRATE)
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
NUM_AGENTS = 1
TRAIN_SEQ_LEN = 100  # take as a train batch
UPDATE_INTERVAL = 100
MODEL_SAVE_INTERVAL = 100
VIDEO_BIT_RATE = Config.BITRATE  # Kbps
HD_REWARD = [1, 2, 3, 12, 15, 20]
BUFFER_NORM_FACTOR = Config.CLIENT_MAX_BUFFER_LEN + 1
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
QUALTITY_COEF = 5
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
FREEZE_PENALTY = 20
LATENCY_PENALTY = 0.2
JUMP_PENALTY = 2
SMOOTH_PENALTY = 3.5
DEFAULT_QUALITY = Config.INITIAL_RATE  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
SUMMARY_DIR = './results'
PENSIEVE_LOG_FILE = './results/pensieve/log'
STALLION_LOG_FILE = './results/stallion/log'
TEST_LOG_FOLDER = './test_results/'
TRAIN_TRACES = './data/cooked_traces/'

CRITIC_MODEL= './results/critic.pt'
ACTOR_MODEL = './results/actor.pt'
CRITIC_MODEL = None

TOTALEPOCH=30000
IS_CENTRAL=True
NO_CENTRAL=False

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor


class Simulator:
    
    def __init__(self, algo):
        self.client = Client(args.ip, args.port, args.algo)
        self.algo = algo
        
    def start(self):
        self.client.register()
        self.client.start()
        
    def stallionRun(self):
        self.solver = stallion.stallion_solver(Config.INITIAL_LATENCY)
        
        with open(STALLION_LOG_FILE + '_record', 'w') as log_file:
            for i in range(610):
                self.solver.update_bw_latency(self.client.bw, self.client.latency)
                rate, _ = self.solver.solve(self.client.get_buffer_size(), self.client.latency)
                latency, idle, buffer_size, freeze, download_time, bw, jump, server_time = self.client.download(rate)
            log_file.write(str(server_time) + '\t' +
                        str(rate) + '\t' +
                        str(bw) + '\t' +
                        str(buffer_size) + '\t' +
                        str(freeze) + '\t' +
                        str(idle) + '\t' +
                        str(latency) + '\t' +
                        str(jump) + '\t')
            log_file.flush()
            
            
    def pensieveRun(self):
        logging.basicConfig(filename=PENSIEVE_LOG_FILE + '_central',
                        filemode='w',
                        level=logging.INFO)
    
        with open(PENSIEVE_LOG_FILE + '_record', 'w') as log_file, open(PENSIEVE_LOG_FILE + '_test', 'w') as test_log_file:

            model_actor = ac.Actor(A_DIM).type(dtype)
            model_critic = ac.Critic(A_DIM).type(dtype)

            model_actor.train()
            model_critic.train()

            optimizer_actor = torch.optim.RMSprop(model_actor.parameters(), lr=ACTOR_LR_RATE)
            optimizer_critic = torch.optim.RMSprop(model_critic.parameters(), lr=CRITIC_LR_RATE)

            # max_grad_norm = MAX_GRAD_NORM 

            state = np.zeros((S_INFO,S_LEN))
            state = torch.from_numpy(state)
            last_rate = DEFAULT_QUALITY
            rate = DEFAULT_QUALITY
            # action_vec = np.zeros(A_DIM)
            # action_vec[rate] = 1
            state[0, -1] = rate / float(np.max(VIDEO_BIT_RATE))  # last quality
            state[1, -1] = float(self.client.bw)
            state[2, -1] = float(self.client.download_time)
            state[3, -1] = float(self.client.latency) / BUFFER_NORM_FACTOR
            state[4, -1] = float(self.client.freeze)  # mega byte
            state[5, -1] = float(self.client.idle)
            state[6, -1] = self.client.get_buffer_size() / BUFFER_NORM_FACTOR  # 10 sec

            done = True
            epoch = 0
            time_stamp = 0

            exploration_size = 4
            episode_steps = 10 ############ testing!!!!!!
            update_num = 1
            batch_size = 20
            gamma = 0.99
            gae_param = 0.95
            clip = 0.2
            ent_coeff = 0.98
            memory = replay_memory.ReplayMemory(exploration_size * episode_steps)
            # memory = ReplayMemory()

            for epoch in range(TOTALEPOCH):
                
                total_bw = 0

                for explore in range(exploration_size):
                    states = []
                    actions = []
                    rewards_comparison = []
                    rewards = []
                    values = []
                    returns = []
                    advantages = []

                    for step in range(episode_steps):

                        prob = model_actor(state.unsqueeze(0).type(dtype))
                        action = prob.multinomial(num_samples=1).detach()
                        v = model_critic(state.unsqueeze(0).type(dtype)).detach().cpu()
                        values.append(v)

                        rate = VIDEO_BIT_RATE[int(action.squeeze().cpu().numpy())]

                        actions.append(torch.tensor([action]))
                        states.append(state.unsqueeze(0))

                        latency, idle, buffer_size, freeze, download_time, bw, jump, server_time = self.client.download(rate)
                        
                        time_stamp = server_time

                        # -- log scale reward --
                        log_rate = np.log(rate)
                        log_last_rate = np.log(last_rate)

                        reward =  QUALTITY_COEF  * log_rate \
                                - FREEZE_PENALTY * freeze \
                                - LATENCY_PENALTY* latency \
                                - JUMP_PENALTY   * jump \
                                - SMOOTH_PENALTY * np.abs(log_rate - log_last_rate) 
                                
                        # reward_max = 2.67
                        # print(f"Get reward: {reward}, log_rate: {log_rate}, freeze: {freeze}, latency: {latency}")
                        # reward = float(max(min(reward, reward_max), -4*reward_max) / reward_max)
                        rewards.append(reward)
                        rewards_comparison.append(torch.tensor([reward]))

                        last_rate = rate
                        total_bw += bw

                        # dequeue history record
                        state = np.roll(state, -1, axis=1)

                        # this should be S_INFO number of terms
                        state[0, -1] = rate / float(np.max(VIDEO_BIT_RATE))  # last quality
                        state[1, -1] = float(bw)
                        state[2, -1] = float(download_time)
                        state[3, -1] = float(latency) / BUFFER_NORM_FACTOR
                        state[4, -1] = float(freeze)  # mega byte
                        state[5, -1] = float(idle)
                        state[6, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec

                        state = torch.from_numpy(state)

                    # log time_stamp, rate, buffer_size, reward
                        log_file.write(str(time_stamp) + '\t' +
                                    str(rate) + '\t' +
                                    str(bw) + '\t' +
                                    str(buffer_size) + '\t' +
                                    str(freeze) + '\t' +
                                    str(idle) + '\t' +
                                    str(latency) + '\t' +
                                    str(jump) + '\t' +
                                    str(reward) + '\n')
                        log_file.flush()

                    # one last step
                    R = torch.zeros(1, 1)
                    v = model_critic(state.unsqueeze(0).type(dtype)).detach().cpu()
                    R = v.data
                    #================================End of an ep========================================
                    # compute returns and GAE(lambda) advantages:
                    values.append(Variable(R))
                    R = Variable(R)
                    A = Variable(torch.zeros(1, 1))
                    for i in reversed(range(len(rewards))):
                        td = rewards[i] + gamma * values[i + 1].data[0, 0] - values[i].data[0, 0]
                        A = float(td) + gamma * gae_param * A
                        advantages.insert(0, A)
                        # R = A + values[i]
                        R = gamma * R + rewards[i]
                        returns.insert(0, R)
                    # store usefull info:
                    # memory.push([states[1:], actions[1:], rewards_comparison[1:], returns[1:], advantages[1:]])
                    memory.push([states, actions, returns, advantages])
            
                # policy grad updates:
                model_actor_old = ac.Actor(A_DIM).type(dtype)
                model_actor_old.load_state_dict(model_actor.state_dict())
                model_critic_old = ac.Critic(A_DIM).type(dtype)
                model_critic_old.load_state_dict(model_critic.state_dict())

                ## actor update
                print("Start training...")
                for update_step in range(update_num):
                    model_actor.zero_grad()
                    model_critic.zero_grad()

                    # new mini_batch
                    # priority_batch_size = int(memory.get_capacity()/10)
                    batch_states, batch_actions, batch_returns, batch_advantages = memory.sample(batch_size)
                    # batch_size = memory.return_size()
                    # batch_states, batch_actions, batch_returns, batch_advantages = memory.pop(batch_size)

                    # old_prob
                    probs_old = model_actor_old(batch_states.type(dtype).detach())
                    v_pre_old = model_critic_old(batch_states.type(dtype).detach())
                    prob_value_old = torch.gather(probs_old, dim=1, index=batch_actions.unsqueeze(1).type(dlongtype))

                    # new prob
                    probs = model_actor(batch_states.type(dtype))
                    v_pre = model_critic(batch_states.type(dtype))
                    prob_value = torch.gather(probs, dim=1, index=batch_actions.unsqueeze(1).type(dlongtype))

                    # ratio
                    ratio = prob_value / (1e-6 + prob_value_old)

                    ## non-clip loss
                    # surrogate_loss = ratio * batch_advantages.type(dtype)


                    # clip loss
                    surr1 = ratio * batch_advantages.type(dtype)  # surrogate from conservative policy iteration
                    surr2 = ratio.clamp(1 - clip, 1 + clip) * batch_advantages.type(dtype)
                    loss_clip_actor = -torch.mean(torch.min(surr1, surr2))
                    # value loss
                    vfloss1 = (v_pre - batch_returns.type(dtype)) ** 2
                    v_pred_clipped = v_pre_old + (v_pre - v_pre_old).clamp(-clip, clip)
                    vfloss2 = (v_pred_clipped - batch_returns.type(dtype)) ** 2
                    loss_value = 0.5 * torch.mean(torch.max(vfloss1, vfloss2))
                    # entropy
                    loss_ent = ent_coeff * torch.mean(probs * torch.log(probs + 1e-6))
                    # total
                    policy_total_loss = loss_clip_actor + loss_ent

                    # update 
                    optimizer_actor.zero_grad()
                    optimizer_critic.zero_grad()
                    policy_total_loss.backward()
                    # loss_clip_actor.backward(retain_graph=True)
                    loss_value.backward()
                    optimizer_actor.step()
                    optimizer_critic.step()

                ## test and save the model
                memory.clear()
                logging.info('Epoch: ' + str(epoch) +
                            ' Avg_policy_loss: ' + str(loss_clip_actor.detach().cpu().numpy()) +
                            ' Avg_value_loss: ' + str(loss_value.detach().cpu().numpy()) +
                            ' Avg_entropy_loss: ' + str(A_DIM * loss_ent.detach().cpu().numpy()) +
                            ' Avg_throughput: ' + str(total_bw / (episode_steps * exploration_size)))

                if epoch % UPDATE_INTERVAL == 0:
                    logging.info("Model saved in file")
                    add_str = 'ppo'
                    actor_model_save_path = "./models/pensieve/%s_%s_%d_actor.model" %(str('abr'), add_str, int(epoch))
                    critic_model_save_path = "./models/pensieve/%s_%s_%d_critic.model" %(str('abr'), add_str, int(epoch))
                    torch.save(model_actor.state_dict(), actor_model_save_path)
                    torch.save(model_critic.state_dict(), critic_model_save_path)
                    # entropy_weight = 0.95 * entropy_weight
                    ent_coeff = 0.95 * ent_coeff

if __name__ == '__main__':
    # time.sleep(args.sleep)
    delete_files_in_folder('data/')
    sim = Simulator(args.algo)

    sim.start()
    sim.pensieveRun()
    # sim.stallionRun()