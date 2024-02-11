import logging
import torch
import sys
import numpy as np
from torch.autograd import Variable
from algorithms.components import ac, replay_memory
sys.path.append("..")
from utils.config import Config



S_INFO = 8  # rate, bw, downloading time, latency, freeze, idle, buffer, suggestion
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
QUALTITY_COEF = 3
FREEZE_PENALTY = 30
LATENCY_PENALTY = 0.5
JUMP_PENALTY = 2
SMOOTH_PENALTY = 5
INSTRUCTION_REWARD = 6
DEFAULT_QUALITY = Config.INITIAL_RATE  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
FAIRNESS_COEF = 0.8

SUMMARY_DIR = './results'
PENSIEVE_LOG_FILE = './results/pensieve/log'
STALLION_LOG_FILE = './results/stallion/log'
MARL_LOG_FILE = './results/marl/log'
TEST_LOG_FOLDER = './test_results/'
TRAIN_TRACES = './data/cooked_traces/'

CRITIC_MODEL= './results/critic.pt'
ACTOR_MODEL = './results/actor.pt'
CRITIC_MODEL = None

TOTALEPOCH=50


USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor


class marl_solver:
    
    def __init__(self, client, weight=1):
        self.client = client
        self.weight = weight
        self.state = np.zeros((S_INFO,S_LEN))
        self.state = torch.from_numpy(self.state)
        
    def getState(self, rate, bw, download_time, latency, freeze, idle, buffer_size, instruction):
        # dequeue history record
        self.state = np.roll(self.state, -1, axis=1)

        # this should be S_INFO number of terms
        self.state[0, -1] = rate / float(np.max(VIDEO_BIT_RATE))  # last quality
        self.state[1, -1] = float(bw)
        # self.state[2, -1] = float(download_time)
        self.state[3, -1] = float(latency) / BUFFER_NORM_FACTOR
        self.state[4, -1] = float(freeze)  # mega byte
        self.state[5, -1] = float(idle)
        self.state[6, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
        self.state[7, -1] = instruction  # suggestion
        
        self.state = torch.from_numpy(self.state)
        
        return self.state
        
    def getReward(self, rate, last_rate, freeze, latency, jump, fair_coef, last_instruction, fair_bw, reward_file):
        # -- log scale reward --
        log_rate = np.log(rate)
        log_last_rate = np.log(last_rate)
        log_fair_bw = np.log(fair_bw)
        
        reward_self =  QUALTITY_COEF  * log_rate \
                - FREEZE_PENALTY * freeze \
                - LATENCY_PENALTY* latency \
                - JUMP_PENALTY   * jump \
                - SMOOTH_PENALTY * np.abs(log_rate - log_last_rate) \
                
        reward = fair_coef * reward_self + (1 - fair_coef) * (QUALTITY_COEF * log_rate + min(0, INSTRUCTION_REWARD * last_instruction * (log_rate - log_fair_bw)))
        

        reward_file.write(str(reward_self) + '\t' +
                    str(fair_coef) + '\t' +
                    str(min(0, INSTRUCTION_REWARD * last_instruction * (log_rate - log_fair_bw))) + '\t' +
                    str(reward - fair_coef * reward_self) + '\n'
                    )
        reward_file.flush()
        
        return reward
                        
    
    def solve(self, train=True):
        logging.basicConfig(filename=MARL_LOG_FILE + '_central',
                        filemode='w',
                        level=logging.INFO)
    
        with open(MARL_LOG_FILE + '_record', 'w') as log_file, open(MARL_LOG_FILE + '_reward', 'w') as reward_file:

            model_actor = ac.MActor(A_DIM).type(dtype)
            model_critic = ac.MCritic(A_DIM).type(dtype)

            model_actor.train()
            model_critic.train()

            optimizer_actor = torch.optim.RMSprop(model_actor.parameters(), lr=ACTOR_LR_RATE)
            optimizer_critic = torch.optim.RMSprop(model_critic.parameters(), lr=CRITIC_LR_RATE)

            # max_grad_norm = MAX_GRAD_NORM 

            
            last_rate = DEFAULT_QUALITY
            last_instruction = 0
            rate = DEFAULT_QUALITY
            # action_vec = np.zeros(A_DIM)
            # action_vec[rate] = 1
            self.getState(rate, self.client.bw, self.client.download_time, self.client.latency, self.client.freeze, self.client.idle, self.client.get_buffer_size(), 0)

            done = True
            epoch = 0
            time_stamp = 0

            exploration_size = 8
            episode_steps = 50 ############ testing!!!!!!
            update_num = 32
            batch_size = 32
            gamma = 0.99
            gae_param = 0.95
            clip = 0.2
            ent_coeff = 0.98
            instruction = 0
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

                        prob = model_actor(self.state.unsqueeze(0).type(dtype))
                        action = prob.multinomial(num_samples=1).detach()
                        v = model_critic(self.state.unsqueeze(0).type(dtype)).detach().cpu()
                        values.append(v)

                        rate = VIDEO_BIT_RATE[int(action.squeeze().cpu().numpy())]

                        actions.append(torch.tensor([action]))
                        states.append(self.state.unsqueeze(0))

                        latency, idle, buffer_size, freeze, download_time, bw, jump, server_time, instruction, fair_bw, fair_coef = self.client.download(rate, "PENSIEVE")
                        
                        time_stamp = server_time
                        # print(fair_coef)
                        # get reward
                        reward = self.getReward(rate, last_rate, freeze, latency, jump, fair_coef, last_instruction, fair_bw, reward_file)
                        # print(reward)
                                
                        # reward_max = 2.67
                        # print(f"Get reward: {reward}, log_rate: {log_rate}, freeze: {freeze}, latency: {latency}")
                        # reward = float(max(min(reward, reward_max), -4*reward_max) / reward_max)
                        rewards.append(reward)
                        rewards_comparison.append(torch.tensor([reward]))

                        last_rate = rate
                        last_instruction = instruction
                        total_bw += bw
                        
                        # state = torch.from_numpy(state)
                        self.getState(rate, bw, download_time, latency, freeze, idle, buffer_size, instruction)

                    # log time_stamp, rate, buffer_size, reward
                        log_file.write(str(time_stamp) + '\t' +
                                    str(rate) + '\t' +
                                    str(bw) + '\t' +
                                    str(buffer_size) + '\t' +
                                    str(freeze) + '\t' +
                                    str(idle) + '\t' +
                                    str(latency) + '\t' +
                                    str(jump) + '\t' +
                                    str(reward) + '\t' + 
                                    str(instruction) + '\t' +
                                    str(fair_bw) + '\t' +
                                    str(fair_coef) + '\n')
                        log_file.flush()

                    # one last step
                    R = torch.zeros(1, 1)
                    v = model_critic(self.state.unsqueeze(0).type(dtype)).detach().cpu()
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
                model_actor_old = ac.MActor(A_DIM).type(dtype)
                model_actor_old.load_state_dict(model_actor.state_dict())
                model_critic_old = ac.MCritic(A_DIM).type(dtype)
                model_critic_old.load_state_dict(model_critic.state_dict())

                ## actor update
                print("Start training...")
                if train:
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
                        actor_model_save_path = "./models/marl/%s_%s_%d_actor.model" %(str('abr'), add_str, int(epoch))
                        critic_model_save_path = "./models/marl/%s_%s_%d_critic.model" %(str('abr'), add_str, int(epoch))
                        torch.save(model_actor.state_dict(), actor_model_save_path)
                        torch.save(model_critic.state_dict(), critic_model_save_path)
                        # entropy_weight = 0.95 * entropy_weight
                        ent_coeff = 0.95 * ent_coeff