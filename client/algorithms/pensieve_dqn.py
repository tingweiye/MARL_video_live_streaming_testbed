import logging
import torch
import sys
import numpy as np
import threading
from torch.autograd import Variable
import torch.optim as optim
from algorithms.components.ddqn import ddqn, OptimizerSpec
sys.path.append("..")
from utils.config import Config
from utils.utils import Logger


S_INFO = 4  
S_LEN = 10  # take how many gops in the past
# A_DIM = len(Config.BITRATE)
# ACTOR_LR_RATE = 0.0001
# CRITIC_LR_RATE = 0.001
# NUM_AGENTS = 1
# TRAIN_SEQ_LEN = 100  # take as a train batch
# MODEL_SAVE_INTERVAL = 50
# VIDEO_BIT_RATE = Config.BITRATE  # Kbps
# HD_REWARD = [1, 2, 3, 12, 15, 20]
# BUFFER_NORM_FACTOR = Config.CLIENT_MAX_BUFFER_LEN + 1
# CHUNK_TIL_VIDEO_END_CAP = 48.0
# M_IN_K = 1000.0

# QUALTITY_COEF = 10
# FREEZE_PENALTY = 20
# LATENCY_PENALTY = 1
# JUMP_PENALTY = 2
# SMOOTH_PENALTY = 10
VIDEO_BIT_RATE = Config.BITRATE  # Kbps
MAX_RATE = float(np.max(VIDEO_BIT_RATE))
NUM_ACTIONS = len(VIDEO_BIT_RATE)
INITIAL_RATE = VIDEO_BIT_RATE[int(len(VIDEO_BIT_RATE) // 2)]

MAX_EPSILON_C = 0.9
MAX_EPSILON_M = 0.9
MIN_EPSILON = 0.01
EPSILON_DECAY = 0.995

QUALTITY_COEF = 5
FREEZE_PENALTY = 50
LATENCY_PENALTY = 1
JUMP_PENALTY = 2
SMOOTH_PENALTY = 5

TRAIN_START_LOCAL = 3000
TRAIN_INTERVAL = 100
TRAIN_TIMES = 70
MODEL_SAVE_INTERVAL = 5000

# MAX_EP_LEN = 10
# TRAIN_START_LOCAL = 5
# TRAIN_INTERVAL = 5
# TRAIN_TIMES = 50
# MODEL_SAVE_INTERVAL = 1

DECAY_INTERVAL = 30

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

TOTALEPOCH=160000


USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor


class pensieve_solver:
    
    def __init__(self, client, trail):
        self.client = client
        self.trail = trail
        optimizer_spec = OptimizerSpec(
            constructor=optim.RMSprop,
            kwargs=dict(lr=0.00025, alpha=0.95, eps=0.01),
        )
        self.agent = ddqn(optimizer_spec)
        
    def train_local_controller(self):
        Logger.log("Training local controller...")
        for _ in range(TRAIN_TIMES):
            self.agent.update_controller()
        Logger.log("Local controller training completed")
        
    def select_rate(self, state_goal, epsilon):
        action = self.agent.select_action(torch.from_numpy(state_goal).unsqueeze(0).type(dtype), epilson=epsilon)
        self.agent.local_count += 1
        return VIDEO_BIT_RATE[action.item()], action
    
    def solve(self, train=True):
        with open(PENSIEVE_LOG_FILE + '_record_' + str(self.trail), 'w') as log_file:

            # max_grad_norm = MAX_GRAD_NORM 

            state = np.zeros((S_INFO,S_LEN))
            last_state = np.zeros((S_INFO,S_LEN))
            # state = torch.from_numpy(state)
            last_rate = DEFAULT_QUALITY
            rate = DEFAULT_QUALITY
            # action_vec = np.zeros(A_DIM)
            # action_vec[rate] = 1
            state[0, -1] = rate / float(np.max(VIDEO_BIT_RATE))  # last quality
            state[1, -1] = float(self.client.bw) / 10
            state[2, -1] = float(self.client.idle)
            state[3, -1] = self.client.get_buffer_size() / 10  # 10 sec

            done = True
            epoch = 0
            time_stamp = 0

            exploration_size = 10
            episode_steps = 30 
            update_num = 20
            batch_size = 32
            gamma = 0.95
            gae_param = 0.95
            clip = 0.2
            ent_coeff = 0.98
            epsilon = 0.9
            # memory = ReplayMemory()

            for epoch in range(TOTALEPOCH):
                rate, rate_idx = self.select_rate(state, epsilon)
                Logger.log(f"Get rate {rate} with epsilon {epsilon}")

                info = self.client.download(rate, "PENSIEVE")

                latency = info["latency"]
                idle = info["idle"]
                buffer_size = info["buffer_size"]
                freeze = info["freeze"]
                download_time = info["download_time"]
                bw = info["bw"]
                jump = info["jump"]
                server_time = info["server_time"]
                true_bandwidth = info["true_bandwidth"]
                propotional_fairness = info["propotional_fairness"]
                maxmin_fairness = info["maxmin_fairness"]
                client_qoe = info["client_qoe"]
                
                time_stamp = server_time

                # -- log scale reward --
                log_rate = np.log(rate)
                log_last_rate = np.log(last_rate)

                reward =  QUALTITY_COEF  * log_rate \
                        - (FREEZE_PENALTY * max(0.75, freeze) if freeze > 0.001 else 0) \
                        - LATENCY_PENALTY* latency \
                        - JUMP_PENALTY   * jump \
                        - SMOOTH_PENALTY * np.abs(log_rate - log_last_rate) 
                # reward_max = 2.67
                # print(f"Get reward: {reward}, log_rate: {log_rate}, freeze: {freeze}, latency: {latency}")
                # reward = float(max(min(reward, reward_max), -4*reward_max) / reward_max)

                last_rate = rate
                last_state = state.copy()
                # dequeue history record
                state = np.roll(state, -1, axis=1)

                # this should be S_INFO number of terms
                state[0, -1] = rate / float(np.max(VIDEO_BIT_RATE))  # last quality
                state[1, -1] = float(np.mean(self.client.bw_his[-10:])) / 10
                state[2, -1] = float(idle)
                state[3, -1] = buffer_size / 10  # 10 sec

                for i in range(100):
                    self.agent.ctrl_replay_memory.push(last_state, rate_idx, state, reward, False)

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
                            str(true_bandwidth) + '\t' +
                            str(propotional_fairness) + '\t' +
                            str(maxmin_fairness) + '\t' +
                            str(client_qoe) + '\n')
                log_file.flush()


                if epoch >= TRAIN_START_LOCAL and epoch % DECAY_INTERVAL == 0:
                    epsilon *= EPSILON_DECAY
                # Save model
                if self.agent.local_count >= TRAIN_START_LOCAL and self.agent.local_count % MODEL_SAVE_INTERVAL == 0:
                    Logger.log("Local controller model saved")
                    self.agent.save_controller_model(self.agent.local_count)
                ## Train local
                if train and epoch >= TRAIN_START_LOCAL and epoch % TRAIN_INTERVAL == 0:
                    print("Start training...")
                    threading.Thread(target=self.train_local_controller).start()
                    # self.train_local_controller()