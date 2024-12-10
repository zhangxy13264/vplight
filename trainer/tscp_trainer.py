import os
import numpy as np
from common.metrics import Metrics
from environment import TSCEnv
from common.registry import Registry
from trainer.base_trainer import BaseTrainer


@Registry.register_trainer("tsc_p")
class TSCPTrainer(BaseTrainer):
    '''
    Register TSCPTrainer for pedestrian-extended traffic signal control tasks.
    '''
    def __init__(self, logger, gpu=0, cpu=False, name="tsc-p"):
        super().__init__(logger=logger,gpu=gpu,cpu=cpu,name=name)
        self.episodes = Registry.mapping['trainer_mapping']['setting'].param['episodes']
        self.steps = Registry.mapping['trainer_mapping']['setting'].param['steps']
        self.test_steps = Registry.mapping['trainer_mapping']['setting'].param['test_steps']
        self.buffer_size = Registry.mapping['trainer_mapping']['setting'].param['buffer_size']
        self.action_interval = Registry.mapping['trainer_mapping']['setting'].param['action_interval']
        self.save_rate = Registry.mapping['logger_mapping']['setting'].param['save_rate']
        self.learning_start = Registry.mapping['trainer_mapping']['setting'].param['learning_start']
        self.update_model_rate = Registry.mapping['trainer_mapping']['setting'].param['update_model_rate']
        self.update_target_rate = Registry.mapping['trainer_mapping']['setting'].param['update_target_rate']
        self.test_when_train = Registry.mapping['trainer_mapping']['setting'].param['test_when_train']
        # self.test_when_train_rate = Registry.mapping['trainer_mapping']['setting'].param['test_when_train_rate']
        # self.pfe_learning_start = Registry.mapping['trainer_mapping']['setting'].param['pfe_learning_start']
        # self.pfe_learning_rate = Registry.mapping['trainer_mapping']['setting'].param['pfe_learning_rate']
        self.PFE = True if Registry.mapping['command_mapping']['setting'].param['agent'] == "vplight" else False

        self.dataset = Registry.mapping['dataset_mapping'][Registry.mapping['command_mapping']['setting'].param['dataset']](
            os.path.join(Registry.mapping['logger_mapping']['path'].path,
                         Registry.mapping['logger_mapping']['setting'].param['data_dir'])
        )
        self.dataset.initiate(ep=self.episodes, step=self.steps, interval=self.action_interval)
        self.yellow_time = Registry.mapping['trainer_mapping']['setting'].param['yellow_length']
        # consists of path of output dir + log_dir + file handlers name
        self.log_file = os.path.join(Registry.mapping['logger_mapping']['path'].path,
                                     Registry.mapping['logger_mapping']['setting'].param['log_dir'],
                                     os.path.basename(self.logger.handlers[-1].baseFilename).rstrip('_BRF.log') + '_DTL.log'
                                     )
        
    def create_world(self):
        '''
        create_world
        Create world, only support SUMO World

        :param: None
        :return: None
        '''
        # traffic setting is in the world mapping
        self.world = Registry.mapping['world_mapping'][Registry.mapping['command_mapping']['setting'].param['world']](
            self.path, Registry.mapping['command_mapping']['setting'].param['thread_num'],interface=Registry.mapping['command_mapping']['setting'].param['interface'])

    def create_metrics(self):
        '''
        create_metrics
        Create metrics to evaluate model performance, currently support reward, queue length, delay(approximate or real) and throughput.

        :param: None
        :return: None
        '''
        if Registry.mapping['command_mapping']['setting'].param['delay_type'] == 'apx':
            lane_metrics = ['rewards', 'queue', 'delay']
            lane_metrics += ['pedestrian_queue', 'pedestrian_delay']
            world_metrics = ['real avg travel time', 'throughput']
            world_metrics += ['real avg pedestrian travel time', 'pedestrian_throughput']
        else:
            lane_metrics = ['rewards', 'queue']
            lane_metrics += ['pedestrian_queue']
            world_metrics = ['delay', 'real avg travel time', 'throughput']
            world_metrics += ['pedestrian_delay', 'real avg pedestrian travel time', 'pedestrian_throughput']
        self.metric = Metrics(lane_metrics, world_metrics, self.world, self.agents)

    def create_agents(self):
        '''
        create_agents
        Create agents for pedestrian-extended traffic signal control tasks.

        :param: None
        :return: None
        '''
        self.agents = []
        agent = Registry.mapping['model_mapping'][Registry.mapping['command_mapping']['setting'].param['agent']](self.world, 0)
        print(agent)
        num_agent = int(len(self.world.intersections) / agent.sub_agents)
        self.agents.append(agent)  # initialized N agents for traffic light control 为每一个路口创建一个agent，i是路口index
        for i in range(1, num_agent):
            self.agents.append(Registry.mapping['model_mapping'][Registry.mapping['command_mapping']['setting'].param['agent']](self.world, i))

        # for magd agents should share information 
        if Registry.mapping['model_mapping']['setting'].param['name'] == 'magd':
            for ag in self.agents:
                ag.link_agents(self.agents)

    def create_env(self):
        '''
        create_env
        Create simulation environment for communication with agents.

        :param: None
        :return: None
        '''
        # TODO: finalized list or non list
        self.env = TSCEnv(self.world, self.agents, self.metric)


    def train(self):
        '''
        train
        Train the agent(s).

        :param: None
        :return: None
        '''
        total_decision_num = 0
        flush = 0
        for e in range(self.episodes):
            # TODO: check this reset agent
            self.metric.clear()
            last_obs = self.env.reset()  # agent * [sub_agent, feature]

            for a in self.agents:
                a.reset()
            if Registry.mapping['command_mapping']['setting'].param['world'] == 'cityflow':
                if self.save_replay and e % self.save_rate == 0:
                    self.env.eng.set_save_replay(True)
                    self.env.eng.set_replay_file(os.path.join(self.replay_file_dir, f"episode_{e}.txt"))
                else:
                    self.env.eng.set_save_replay(False)
            episode_loss = []
            i = 0
            while i < self.steps:
                if i % self.action_interval == 0:
                    last_phase = np.stack([ag.get_phase() for ag in self.agents])  # [agent, intersections]

                    if total_decision_num > self.learning_start:
                        actions = []
                        for idx, ag in enumerate(self.agents):
                            actions.append(ag.get_action(last_obs[idx], last_phase[idx], test=False))                            
                        actions = np.stack(actions)  # [agent, intersections]
                    else:
                        actions = np.stack([ag.sample() for ag in self.agents])

                    actions_prob = []
                    for idx, ag in enumerate(self.agents):
                        actions_prob.append(ag.get_action_prob(last_obs[idx], last_phase[idx]))
                        
                    rewards_list = []
                    for _ in range(self.action_interval):
                        obs, rewards, dones, _ = self.env.step(actions.flatten())
                        i += 1
                        rewards_list.append(np.stack(rewards))
                    rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
                    self.metric.update(rewards)

                    cur_phase = np.stack([ag.get_phase() for ag in self.agents])
                    for idx, ag in enumerate(self.agents):
                        ag.remember(last_obs[idx], last_phase[idx], actions[idx], actions_prob[idx], rewards[idx],
                            obs[idx], cur_phase[idx], dones[idx], f'{e}_{i//self.action_interval}_{ag.id}')

                    flush += 1
                    if flush == self.buffer_size - 1:
                        flush = 0
                        # self.dataset.flush([ag.replay_buffer for ag in self.agents])
                    total_decision_num += 1
                    last_obs = obs
                
                # if self.PFE and total_decision_num > self.pfe_learning_start and total_decision_num % self.pfe_learning_rate == self.pfe_learning_rate - 1:
                #     loss_pfe = np.stack([ag.train_pfe() for ag in self.agents])

                if total_decision_num > self.learning_start and\
                        total_decision_num % self.update_model_rate == self.update_model_rate - 1:
                    cur_loss_q = np.stack([ag.train() for ag in self.agents])  # TODO: training
                    episode_loss.append(cur_loss_q)

                if total_decision_num > self.learning_start and \
                        total_decision_num % self.update_target_rate == self.update_target_rate - 1:
                    [ag.update_target_network() for ag in self.agents]

                if all(dones):
                    break
            if len(episode_loss) > 0:
                mean_loss = np.mean(np.array(episode_loss))
            else:
                mean_loss = 0
            
            self.writeLog("TRAIN", e, self.metric.real_average_travel_time(), self.metric.real_average_pedestrian_travel_time(), self.metric.combine_average_travel_time(),\
                mean_loss, self.metric.rewards(), self.metric.queue(), self.metric.delay(), self.metric.throughput(),
                self.metric.pedestrian_queue(), self.metric.pedestrian_delay(), self.metric.pedestrian_throughput(),
                self.metric.combine_queue(), self.metric.combine_delay(), self.metric.combine_throughput(),
                )
            self.logger.info("-----------------------------episode:{}/{}-----------------------------".format(e, self.episodes))
            self.logger.info("Train step:{}/{}, q_loss:{}, rewards:{:.2f}".format(i, self.steps, mean_loss, self.metric.rewards()))
            self.logger.info("Vehicle:    queue:{:.2f}, delay:{:.2f}, throughput:{}, avg travel time:{:.2f}".format(
                self.metric.queue(), self.metric.delay(), int(self.metric.throughput()), self.metric.real_average_travel_time()
            ))
            self.logger.info("Pedestrian: queue:{:.2f}, delay:{:.2f}, throughput:{}, avg travel time:{:.2f}".format(
                self.metric.pedestrian_queue(), self.metric.pedestrian_delay(), int(self.metric.pedestrian_throughput()), self.metric.real_average_pedestrian_travel_time()
            ))
            self.logger.info("Combine:    queue:{:.2f}, delay:{:.2f}, throughput:{}, avg travel time:{:.2f}".format(
                self.metric.combine_queue(), self.metric.combine_delay(), int(self.metric.combine_throughput()), self.metric.combine_average_travel_time()
            ))
            self.logger.info("")

            for j in range(len(self.world.intersections)):
                self.logger.debug("intersection:{}, mean_episode_reward:{}, mean_queue:{}".format(j, self.metric.lane_rewards()[j],\
                     self.metric.lane_queue()[j]))
            if e % self.save_rate == 0:
                [ag.save_model(e=e) for ag in self.agents]
            # if self.PFE:
            #     [ag.save_pfe(e=e) for ag in self.agents]
                
            if self.test_when_train and e>=70:
                self.train_test(e)

        [ag.save_model(e=self.episodes) for ag in self.agents]

    def train_test(self, e):
        '''
        train_test
        Evaluate model performance after each episode training process.

        :param e: number of episode
        :return self.metric.real_average_travel_time: travel time of vehicles
        '''
        obs = self.env.reset()
        self.metric.clear()
        for a in self.agents:
            if self.PFE:
                a.reset(mode=True)
            else:
                a.reset()
        for i in range(self.test_steps):
            if i % self.action_interval == 0:
                phases = np.stack([ag.get_phase() for ag in self.agents])
                actions = []
                for idx, ag in enumerate(self.agents):
                    actions.append(ag.get_action(obs[idx], phases[idx], test=True))
                actions = np.stack(actions)
                rewards_list = []
                for _ in range(self.action_interval):
                    obs, rewards, dones, _ = self.env.step(actions.flatten())  # make sure action is [intersection]
                    i += 1
                    rewards_list.append(np.stack(rewards))
                rewards = np.mean(rewards_list, axis=0) 
                self.metric.update(rewards)
            if all(dones):
                break
        
        self.writeLog("TEST", e, self.metric.real_average_travel_time(), self.metric.real_average_pedestrian_travel_time(), self.metric.combine_average_travel_time(),\
            100, self.metric.rewards(),self.metric.queue(),self.metric.delay(), self.metric.throughput(),
            self.metric.pedestrian_queue(), self.metric.pedestrian_delay(), self.metric.pedestrian_throughput(),
            self.metric.combine_queue(), self.metric.combine_delay(), self.metric.combine_throughput(),)
        self.logger.info("Test step:{}/{}, rewards:{:.2f}".format(i, self.steps, self.metric.rewards()))
        self.logger.info("Vehicle:    queue:{:.2f}, delay:{:.2f}, throughput:{}, avg travel time:{:.2f}".format(
            self.metric.queue(), self.metric.delay(), int(self.metric.throughput()), self.metric.real_average_travel_time()
        ))
        self.logger.info("Pedestrian: queue:{:.2f}, delay:{:.2f}, throughput:{}, avg travel time:{:.2f}".format(
            self.metric.pedestrian_queue(), self.metric.pedestrian_delay(), int(self.metric.pedestrian_throughput()), self.metric.real_average_pedestrian_travel_time()
        ))
        self.logger.info("Combine:    queue:{:.2f}, delay:{:.2f}, throughput:{}, avg travel time:{:.2f}".format(
            self.metric.combine_queue(), self.metric.combine_delay(), int(self.metric.combine_throughput()), self.metric.combine_average_travel_time()
        ))
        self.logger.info("")
        return self.metric.real_average_travel_time()

    def test(self, drop_load=True):
        '''
        test
        Test process. Evaluate model performance.

        :param drop_load: decide whether to load pretrained model's parameters
        :return self.metric: including queue length, throughput, delay and travel time
        '''
        if Registry.mapping['command_mapping']['setting'].param['world'] == 'cityflow':
            if self.save_replay:
                self.env.eng.set_save_replay(True)
                self.env.eng.set_replay_file(os.path.join(self.replay_file_dir, f"final.txt"))
            else:
                self.env.eng.set_save_replay(False)
        self.metric.clear()
        if not drop_load:
            [ag.load_model(self.episodes) for ag in self.agents]
        obs = self.env.reset()
        for a in self.agents:
            if self.PFE:
                a.reset(mode=True)
            else:
                a.reset()
        for i in range(self.test_steps):
            if i % self.action_interval == 0:
                phases = np.stack([ag.get_phase() for ag in self.agents])
                actions = []
                for idx, ag in enumerate(self.agents):
                    actions.append(ag.get_action(obs[idx], phases[idx], test=True))
                actions = np.stack(actions)
                rewards_list = []
                for j in range(self.action_interval):
                    obs, rewards, dones, _ = self.env.step(actions.flatten())
                    i += 1
                    rewards_list.append(np.stack(rewards))
                rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
                self.metric.update(rewards)
            if all(dones):
                break
        self.logger.info("Final Test: mean rewards:{:.2f}".format(i, self.steps, self.metric.rewards()))
        self.logger.info("Vehicle:    queue:{:.2f}, delay:{:.2f}, throughput:{}, avg travel time:{:.2f}".format(
            self.metric.queue(), self.metric.delay(), int(self.metric.throughput()), self.metric.real_average_travel_time()
        ))
        self.logger.info("Pedestrian: queue:{:.2f}, delay:{:.2f}, throughput:{}, avg travel time:{:.2f}".format(
            self.metric.pedestrian_queue(), self.metric.pedestrian_delay(), int(self.metric.pedestrian_throughput()), self.metric.real_average_pedestrian_travel_time()
        ))
        self.logger.info("Combine:    queue:{:.2f}, delay:{:.2f}, throughput:{}, avg travel time:{:.2f}".format(
            self.metric.combine_queue(), self.metric.combine_delay(), int(self.metric.combine_throughput()), self.metric.combine_average_travel_time()
        ))
        self.logger.info("")
        return self.metric

    def writeLog(self, mode, step, travel_time, pedestrian_travel_time, combine_travel_time, loss, cur_rwd, cur_queue, cur_delay, cur_throughput, cur_pedestrian_queue, cur_pedestrian_delay, cur_pedestrian_throughput, cur_com_queue, cur_com_delay, cur_com_throughput):
        res = Registry.mapping['model_mapping']['setting'].param['name'] + '\t' + mode + '\t' + str(
            step) + '\t' + "%.1f" % travel_time + '\t' + "%.1f" % pedestrian_travel_time + '\t'  + "%.1f" % combine_travel_time + '\t' + "%.1f" % loss + "\t" +\
            "%.2f" % cur_rwd + "\t" + "%.2f" % cur_queue + "\t" + "%.2f" % cur_delay + "\t" + "%d" % cur_throughput + "\t" +\
            "%.2f" % cur_pedestrian_queue + "\t" + "%.2f" % cur_pedestrian_delay + "\t" + "%d" % cur_pedestrian_throughput + "\t" +\
            "%.2f" % cur_com_queue + "\t" + "%.2f" % cur_com_delay + "\t" + "%d" % cur_com_throughput 
        log_handle = open(self.log_file, "a")
        log_handle.write(res + "\n")
        log_handle.close()