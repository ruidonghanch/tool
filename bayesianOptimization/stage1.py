# -*- coding:utf8 -*-
"""
    v0.4 锁定Z轴朝向
"""
import warnings
import os
import pickle as pkl
warnings.filterwarnings('ignore')
import numpy as np
import math
from matplotlib import pyplot as plt
import pygame
from object import *
from bayesian.gaussian import get_next_point
# from ddpg_DDTF import DDPG
import os, time
import copy


class Plate:
    def __init__(self, dt, render=False):  # 初始化参数
        self.frame = 0
        self.time = 0
        self.pause = False
        self.render = render
        if self.render:
            self.screen = screen
        pygame.init()
        self.dt = dt / 1000

        # self.radar1 = Radar(L_radar1_R, self.frame, 1)
        # self.radar2 = Radar(L_radar2_R, self.frame, 2)
        # self.radar3 = Radar(L_radar3_R, self.frame, 3)
        self.missile = Missile(MISSILE_LOC, V_MISSILE)

        self.all_sprite = pygame.sprite.Group()
        self.radar = pygame.sprite.Group()

        for i, r_paras in enumerate(L_radars):
            self.radar.add(Radar(r_paras, self.frame, i))

        self.jams = pygame.sprite.Group()
        self.interceptors = pygame.sprite.Group()
        self.interceptors_max_num = 0

        # self.radar.add(self.radar1, self.radar2, self.radar3)
        self.all_sprite.add(self.missile, self.radar)

        self.anti_detection = {}
        self.anti_detection_overall = None
        self.reset()

    def reset(self):  # 游戏回合重置
        self.time_reset()
        self.score = 0
        self.steps = 0
        self.missile.reset(MISSILE_LOC, V_MISSILE)
        self.all_sprite.remove(self.jams, self.interceptors)
        self.jams.empty()
        self.interceptors.empty()
        self.terminal = False

    def handle_event(self):  # 人机接口,处理输入事件
        active = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                if event.key == pygame.K_SPACE:
                    self.paused = True
                    self._pause()
                if event.key == pygame.K_m and self.missile.jams > 0:
                    active = 1

                # 键盘控制躲避
                if event.key == pygame.K_KP1:
                    a = np.array([-0.5, 0.5, 0])
                    self.missile.change_orbit(a)
                if event.key == pygame.K_KP2:
                    a = np.array([0, 1, 0])
                    self.missile.change_orbit(a)
                if event.key == pygame.K_KP3:
                    a = np.array([0.5, 0.5, 0])
                    self.missile.change_orbit(a)
                if event.key == pygame.K_KP4:
                    a = np.array([-1, 0, 0])
                    self.missile.change_orbit(a)
                if event.key == pygame.K_KP6:
                    a = np.array([1, 0, 0])
                    self.missile.change_orbit(a)
                if event.key == pygame.K_KP7:
                    a = np.array([-0.5, -0.5, 0])
                    self.missile.change_orbit(a)
                if event.key == pygame.K_KP8:
                    a = np.array([0, -1, 0])
                    self.missile.change_orbit(a)
                if event.key == pygame.K_KP9:
                    a = np.array([0.5, -0.5, 0])
                    self.missile.change_orbit(a)
        return active

    def _pause(self):  # 暂停函数
        pause_time = pygame.time.get_ticks()
        while self.paused:
            event = pygame.event.wait()
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                if event.key == pygame.K_SPACE:
                    self.paused = False
        self.pause_time_now = pygame.time.get_ticks() - pause_time

    def getplateState(self):  # 给出状态,状态包含全部信息(位置,剩余干扰机数量,是否被探测到)
        return self.missile.x, self.missile.y, self.missile.z, self.missile.jams, self.anti_detection_overall

    def time_reset(self):  # 时间重置
        pass

    def game_over(self):  # 游戏结束
        if self.missile.x != BASEX and self.missile.z <= 0:
            # print(self.missile.x, ENEMYBASEX, self.missile.y, ENEMYBASEY)
            d_arr = np.array([self.missile.x - ENEMYBASEX, self.missile.y - ENEMYBASEY])
            if self.missile.z < 0:
                pass
            if np.linalg.norm(d_arr) <= 5:
                return 'win'
            else:
                return 'lose'

        for i in self.interceptors:
            d = np.linalg.norm(i.get_missile_pos() - self.missile.get_missile_pos())
            if d < 5:
                return 'lose'

    @staticmethod
    def cal_interceptors_velocity(RL, RV, BL, BVM):
        RL, RV, BL = np.array(RL), np.array(RV), np.array(BL)
        result = []
        a = (RL ** 2).sum() + (BL ** 2).sum() - 2 * (RL * BL).sum()
        b = 2 * (RV * (RL - BL)).sum()
        # print('b', b)
        c = (RV ** 2).sum() - BVM ** 2
        if b:
            n1 = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
            n2 = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
            for i in (n1, n2):
                if i and i > 0:
                    result.append((RL - BL) * i + RV)
        try:
            return result[0]
        except:
            pass

    def step(self, active, action2=None):  # 每步骤更新

        if active:
            jam = self.missile.shoot()
            if jam is not None:
                self.all_sprite.add(jam)
                self.jams.add(jam)

        # 每步更新
        self.handle_event()
        self.all_sprite.update(self.dt)
        if action2 is not None:
            action2 *= V_MISSILE_C
            self.missile.change_orbit(action2)

        # 防空
        distance = np.linalg.norm(self.missile.get_missile_pos() - np.array([ENEMYBASEX, ENEMYBASEY, 0]))
        if distance < 500 and len(self.interceptors) < self.interceptors_max_num:  # todo 可改防空范围   300->253->96步防御
            missile_loc_bias = 300
            v = self.cal_interceptors_velocity(self.missile.get_missile_pos(), self.missile.speed,
                                               [ENEMYBASEX-missile_loc_bias, ENEMYBASEY, 0], 30)

            if v is not None:
                i_missile = Missile([ENEMYBASEX-missile_loc_bias, ENEMYBASEY, 0], v)
                self.all_sprite.add(i_missile)
                self.interceptors.add(i_missile)
            pass

        # 根据底层物理 模型的运算结果计算分数
        for radar in self.radar:
            distance = distance_dd_radar(self.missile.get_missile_pos(), radar.getradarpos())

            if distance > radar.r_radar:
                self.anti_detection[radar.ID] = True  # dd距离大于探测半径

            elif len(self.jams) > 0:
                anti_detection_temp = radar.detection_radar_dd_jam(self.jams, self.missile.get_missile_pos())
                # print('雷达', radar.ID, '干扰状态', anti_detection_temp) #debug测试

                for key in anti_detection_temp:
                    if anti_detection_temp[key] == True:
                        self.anti_detection[radar.ID] = True
                    else:
                        self.anti_detection[radar.ID] = False  # 其余情况均为失败反探测

                        # self.draw_text(self.screen, u"Score:%d" % self.score, FONT_SIZE, 150, 20, WHITE)
            else:
                self.anti_detection[radar.ID] = False

        if False in self.anti_detection.values():
            self.anti_detection_overall = False
        else:
            self.anti_detection_overall = True

        if not self.anti_detection_overall:
            self.score = self.score - 10
        else:
            # self.score = self.score + 10
            pass

        self.terminal = self.game_over()
        if self.render:
            self.draw()
        self.steps += 1

        # 第二阶段的reward设置
        if self.interceptors:
            reward = 0
            # 理解中的稀疏有两个方向，过程中没有所导致的稀疏，和内容上成分少的稀疏
            # reward = -np.linalg.norm(self.missile.get_missile_pos() - [ENEMYBASEX, ENEMYBASEY, 0]) / 500
            # if self.terminal:
            #     reward = -np.linalg.norm(self.missile.get_missile_pos() - [ENEMYBASEX, ENEMYBASEY, 0]) / 10
            if self.terminal == 'win':
                reward += 50
            # elif self.terminal == 'lose':
            #     reward += -20

            info = ''
            # print(state, reward)
            return self.get_state(), reward, self.terminal, info

    def get_state(self):
        state = list(self.trans_arr(self.missile.get_missile_pos(), 'pos'))
        state.extend(list(self.trans_arr(self.missile.speed, 'spd')))
        for i in self.interceptors:
            state.extend(list(self.trans_arr(i.get_missile_pos(), 'pos')))
            state.extend(list(self.trans_arr(i.speed, 'spd')))
        # print(state)
        return state

    @staticmethod
    def trans_arr(arr, style):
        if style == 'pos':
            edge = np.array([WIDTH, HEIGHT, 200])
        elif style == 'spd':
            edge = np.array([50, 50, 50])
        else:
            raise ValueError('style =', style)
        return arr * 2 / edge - 1

    def draw_text(self, surf, text, size, x, y, color):  # 在pygame上显示文字内容
        pygame.font.init()
        font = pygame.font.SysFont('ubuntu', size)
        text_surface = font.render(text, True, color)  # # True denotes the font to be anti-aliased
        text_rect = text_surface.get_rect()
        text_rect.midtop = (x, y)
        surf.blit(text_surface, text_rect)

    def draw(self):  # 在pygame上显示图片内
        self.screen.blit(background, (0, 0))
        # self.screen.blit(image_blast, (0, 0))     # 测试图片
        self.draw_text(self.screen, u"Score:%d" % self.score, FONT_SIZE, 150, 20, WHITE)
        self.draw_text(self.screen, u'Realtime in mins:%s' % self.missile.realtime_mins, FONT_SIZE, 150, 40, WHITE)

        # draw blast
        # draw shadows
        def blit_shadow(r, x, y):
            sub_s = pygame.Surface(r[2:])
            sub_s.set_colorkey((0, 0, 0))
            r = r[:]
            r[3] /= 2
            pygame.draw.ellipse(sub_s, (10, 10, 10), r)
            sub_s.set_alpha(180)
            self.screen.blit(sub_s, (x - r[2] / 2, y + r[3] / 2))

        b_r, r_r, _r = image_base_blue.get_rect(), image_base_red.get_rect(), image_base.get_rect()
        blit_shadow(b_r, ENEMYBASEX, ENEMYBASEY)
        blit_shadow(r_r, BASEX, BASEY)
        blit_shadow(_r, ENEMYBASEX - 300, ENEMYBASEY)

        for i in self.all_sprite:
            rect = i.image_o.get_rect()
            rect[2] = i.image.get_rect()[2]
            blit_shadow(rect, i.x, i.y)

        self.screen.blit(image_base, (ENEMYBASEX - 300 - _r[2] / 2, ENEMYBASEY - _r[3]/2))
        self.screen.blit(image_base_blue, (ENEMYBASEX - b_r[2] / 2, ENEMYBASEY - b_r[3]/2))
        self.screen.blit(image_base_red, (BASEX - r_r[2] / 2, BASEY - r_r[3]/2))
        # pygame.draw.circle(self.screen, (255, 0, 0), (ENEMYBASEX, ENEMYBASEY), 5)
        # pygame.draw.circle(self.screen, (255, 0, 0), (BASEX, BASEY), 5)
        self.all_sprite.draw(self.screen)
        if self.terminal:
            r = image_blast.get_rect()
            self.screen.blit(image_blast, (self.missile.rect.centerx - r[2] / 2, self.missile.rect.centery - r[3] / 2))


class EachEpoch:
    def __init__(self, render=False):
        self.plate = Plate(1000.0 / LOCK_FPS, render)
        self.action_space = 21
        # todo action max, min
        self.action_max = np.array([10, 10, 10, 10, 10, 70])
        self.action_min = np.array([-10, -10, -10, -10, -10, 0])
        assert (self.action_max > self.action_min).all()
        self.action_edge_max = copy.copy(self.action_max)
        self.action_edge_min = copy.copy(self.action_min)
        self.agent = None
        self.epoch = 0
        self.last_total_reward = None

    def sample(self, v_lock=False):
        result = self.action_min + (self.action_max - self.action_min) * np.random.rand(len(self.action_max))
        if v_lock:
            return result[3:]
        else:
            return result

    def run(self, actions1: 'n * len(self.action_max)' = None, test=False, show_info=True, dynamic_edge=False):
        """

        :param actions1: np.shape(actions1) = (-1, 6) or (-1, 3)
        :param test: if test mode only
        :param show_info:
        :param dynamic_edge:
        :return:
        """
        # 输入检查
        if actions1 is not None:
            try:
                a_s_1 = np.shape(actions1)[1]
                assert a_s_1 == len(self.action_max) or a_s_1 == len(self.action_max) - 3
            except Exception as e:
                raise ValueError('input error, your action value is: {}'.format(actions1))
            # 输入情况
            i_bias = 3
            if a_s_1 == len(self.action_max):
                i_bias = 0
        else:
            actions1 = []

        # 阶段1
        selections = {}

        for a in actions1:
            # if dynamic_edge:
            #     for i in range(len(actions1)):
            #         # 均匀扩展
            #         small_step = (self.action_max[i] - self.action_min[i]) * 0.005
            #         if a[i] > self.action_edge_max[i + i_bias]:
            #             print('element out of range')
            #             a[i] = self.action_edge_max[i + i_bias] = self.action_edge_max[i + i_bias] + small_step
            #         elif a[i] < self.action_edge_min[i + i_bias]:
            #             print('element out of range')
            #             a[i] = self.action_edge_min[i + i_bias] = self.action_edge_min[i + i_bias] - small_step

            # 膨胀扩展
            # rate = 1.005
            # if a[i] > self.action_edge_max[i]:
            #     print('element out of range')
            #     a[i] = self.action_edge_max[i] = self.action_edge_max[i+i_bias] * rate
            # elif a[i] < self.action_edge_min[i]:
            #     print('element out of range')
            #     a[i] = self.action_edge_min[i] = self.action_edge_min[i+i_bias] * rate

            # 兼容3输入和6输入
            if i_bias == 0:
                selections[int(a[-1])] = np.array(a[:3]), np.array([a[3], a[4], 2])
            else:
                selections[int(a[-1])] = np.zeros(3), np.array([a[0], a[1], 2])

        # 配置干扰机动作
        timing = list(selections.keys())
        timing.sort()
        for i in range(len(timing)):
            DV_JAMS[i + 1] = selections[timing[i]][0]
            VECTOR_JAMS[i + 1] = selections[timing[i]][1]

        # run steps
        state = None
        reward_sum = 0
        while True:
            act = 1 if self.plate.steps in timing else 0
            t = self.plate.terminal
            self.plate.handle_event()

            if t:
                # print(t)    # print 出胜负                # pygame.time.wait(1000)
                break
            else:
                if self.agent and self.plate.interceptors:
                    state = self.plate.get_state()
                    if not test:
                        NA, TA = self.agent.noise_action(state)
                        state_, reward, terminal, info = self.plate.step(act, NA)
                        self.agent.perceive(state, NA, reward, state_, terminal)
                        if show_info:
                            print('\rTraining... epoch:{} step:{} reward:{} TA:{} NA:{}'.format(
                                self.epoch, self.plate.steps, reward, TA, NA), end='')
                    else:
                        TA = self.agent.action(state)
                        state_, reward, terminal, info = self.plate.step(act, TA)
                        if show_info:
                            print('\rTesting({})... epoch:{} step:{} LTA:{} reward:{} action:{}'.format(
                                test, self.epoch, self.plate.steps, self.last_total_reward, reward, TA), end='')
                    reward_sum += reward
                    # state = state_
                else:
                    self.plate.step(act, None)
                    if show_info:
                        if not test:
                            print('\rTraining... epoch:{} step:{}'.format(self.epoch, self.plate.steps), end='')
                        else:
                            print('\rTesting({})... epoch:{} step:{} LTA:{}'.format(test, self.epoch, self.plate.steps,
                                                                                    self.last_total_reward), end='')
                pygame.display.update()
            # print(self.plate.steps)
        pygame.time.wait(100)
        self.last_total_reward = reward_sum
        return self.plate.score, reward_sum

    def reset(self):
        self.plate.reset()


def dynamic_edge(actions, env):
    """
    动态边界功能
    :param actions:
    :param env:
    :return:
    """

    # 输入检查
    try:
        a_s_1 = np.shape(actions)[1]
        assert a_s_1 == len(env.action_max) or a_s_1 == len(env.action_max) - 3
    except Exception as e:
        raise ValueError('input error, your action value is: {}'.format(actions))

    i_bias = 3
    if a_s_1 == len(env.action_max):
        i_bias = 0

    for a in actions:
        for i in range(len(actions)):
            # 均匀扩展
            small_step = (env.action_max[i] - env.action_min[i]) * 0.005
            if a[i] > env.action_edge_max[i + i_bias]:
                print('element out of range')
                a[i] = env.action_edge_max[i + i_bias] = env.action_edge_max[i + i_bias] + small_step
            elif a[i] < env.action_edge_min[i + i_bias]:
                print('element out of range')
                a[i] = env.action_edge_min[i + i_bias] = env.action_edge_min[i + i_bias] - small_step
    return actions


# if __name__ == "__main__":
#     plate = Plate(screen, 1000.0 / LOCK_FPS)
#
#     step = 0
#     while True:
#         act = plate.handle_event()
#         print(step)
#         s = plate.game_over()
#         if s:
#             print(s)
#             pygame.time.wait(1000)
#             plate.reset()
#             step = 0
#         else:
#             plate.step(act)
#             pygame.display.update()
#         step += 1

class Agent:
    def __init__(self, func):
        self.action_buffer = []
        self.reward_buffer = []
        self.func = func

    def act(self):
        pass

    def learn(self, action, reward):
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)


def create_file_folder(info):
    path = os.path.join('.', 'save')
    if not os.path.exists(path):
        os.makedirs(path)

    t = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
    path_ = os.path.join(path, info + '_' + t)
    while os.path.exists(path_):
        t = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
        path_ = os.path.join(path, info + '_' + t)
    os.makedirs(path_)
    # os.makedirs(os.path.join(path_, 'images'))
    return path_


# test
# if __name__ == "__main__":
#     e = EachEpoch()
#
#     while True:
#         # actions1 = [1, 2, 3, 3, 4, 5, 100, 2, 3, 1, 1, 2, 3, 200, 1, 2, 1, 1, 2, 3, 300]
#         actions1 = [1, 2, 3, 3, 4, 5, 100], [2, 3, 1, 1, 2, 3, 200], [1, 2, 1, 1, 2, 3, 300]
#         score = e.run(actions1)
#         print(score)
#         e.reset()
#         pygame.time.wait(5000)


def sample_func(env: EachEpoch, mode, action: 'n*[action]', sample_num=10, score_diff=True, band=None, ):
    """
    采样框架。将根据model设v_lock然后在环境中采样
    :param env: 环境对象
    :param mode: 模式
    :param action: 已有的action。[]表示当前无动作，表示正在释放第1架无人机
    :param sample_num: 采样数量。需要采样多少个样本
    :param score_diff: 是否确定要不同得分的样本
    :param band: 干扰机朝向小范围变动的界限

    :return: action_buffer, score_buffer    注：action_buffer中的动作只包含采样的动作，不包含已有的动作
    """
    if mode == 1:
        v_lock = True
    elif mode == 2:
        v_lock = False
        assert len(action) >= 1
        assert np.shape(action)[1] == 3
        band = [abs(0.1*x) for x in action[-1]]
    else:
        raise ValueError

    def sample_with_ND(act, band):
        if type(act) == int:
            return np.random.uniform(act - band, act + band)
        else:
            result = []
            for a,b in zip(act,band):
                result.append(np.random.uniform(a - b, a + b))
            return result

    action_buffer, score_buffer = [], []
    while len(score_buffer) < sample_num:
        act = action[:]
        if v_lock:
            act.append(env.sample(v_lock))
        else:
            assert len(act[-1]) == 3
            ang_explore = sample_with_ND(act[-1][:-1], band) + [act[-1][-1]]
            act_sample = list(env.sample())
            act[-1] = np.array(act_sample[:3] + ang_explore)
        env.reset()
        score, total_reward = env.run(np.array(act), show_info=False)
        # print(score, actions1)
        if (score_diff and score not in score_buffer) or not score_diff:
            print('\r' + str(len(score_buffer) + 1), 'sampled', score, act[-1])
            action_buffer.append(act[-1])
            score_buffer.append(score)

    return action_buffer, score_buffer

def save_sample(name, data):
    assert type(name) == str
    if not os.path.exists('save'):
        os.makedirs('save')
    path = os.path.join('save', name + '.txt')
    with open(path, 'wb') as f:
        pkl.dump(data, f)


def load_sample(name):
    assert type(name) == str
    path = os.path.join('save', name + '.txt')
    with open(path, 'rb') as f:
        data = pkl.load(f)
    return data

def bayesian(env: EachEpoch, action, iter = 300):
    '''mode1为角度探索，model2为全局所有变量的探索'''
    best_action = action

    mode = 1
    initial_theta = [0.1,0.1,0.1]
    edgeL = [-2.,-2.,0.]
    edgeU = [2.,2.,70.]
    action_buffer, score_buffer = sample_func(env, mode, best_action, sample_num = 10, score_diff = True)
    for  i in range(iter):
        act,gp_theta = get_next_point(np.array(action_buffer), np.array(score_buffer), iter_num = 2000, edgeL = edgeL, edgeU = edgeU, min_iter= 200, initial_theta=initial_theta, initial_k = 1)

        initial_theta = gp_theta
        act = act[0].tolist()
        this_act = best_action[:]
        this_act.append(act)

        env.reset()
        score, total_reward = env.run(np.array(this_act), show_info=False)
        action_buffer.append(act)
        score_buffer.append(score)
        print('iter', i, 'score', score, 'best_score', max(score_buffer))

        #终止条件设置
        if len(score_buffer) > 100 and np.std(np.array(score_buffer[-20:-1])) < 5:
            break


    # 从buffer中找到最好值
    pos = score_buffer.index(max(score_buffer))
    best_action_lock = best_action[:]

    best_action_lock.append(action_buffer[pos])

    print('start step2,best angle is', action_buffer[pos])

    angle_x = action_buffer[pos][0]
    angle_y = action_buffer[pos][1]
    time = action_buffer[pos][2]

    mode = 2
    action_buffer, score_buffer = sample_func(env, mode, best_action_lock, sample_num=10, score_diff=True,band = 0.5)

    edgeU = [1,1,1,angle_x + abs(angle_x)*0.1,angle_y + abs(angle_y)*0.1,time+2]
    edgeL = [-1,-1,-1,angle_x - abs(angle_x)*0.1,angle_y - abs(angle_y)*0.1,time-2]
    initial_theta = [0.1,0.1,0.1,0.1,0.1,0.1,0.1]
    for i in range(iter):
        act,gp_theta = get_next_point(np.array(action_buffer), np.array(score_buffer), iter_num = 2000, edgeL = edgeL, edgeU = edgeU, min_iter= 200, initial_theta=initial_theta, initial_k = 1)

        act = act[0]
        initial_theta = gp_theta
        this_act = best_action[:]
        this_act.append(act)

        env.reset()
        score, total_reward = env.run(np.array(this_act), show_info=False)
        action_buffer.append(act)
        score_buffer.append(score)
        print('iter', i, 'score', score, 'best_score', max(score_buffer))

        # 终止条件设置
        if len(score_buffer) > 50 and np.std(np.array(score_buffer[-20:-1])) < 20:
            break

    # 从buffer中找到最好值
    pos = score_buffer.index(max(score_buffer))
    best_action.append(action_buffer[pos])

    return best_action

if __name__ == '__main__':
    # act,gp_theta,beyond_edge = get_next_point(np.array(action_buffer), np.array(score_buffer).T, 2000,
    #                      [0.8, 0.8, 0.8, -0.42948767 * 0.93, 0.53134668 * 1.07, 33],
    #                      [-0.8, -0.8, -0.8, -0.42948767 * 1.07, 0.53134668 * 0.93, 31],
    #                      [1, 1, 1, -0.42948767 * 0.5, 0.53134668 * 1.5, 34],
    #                      [-1, -1, -1, -0.42948767 * 1.5, 0.53134668 * 0.5, 30],
    #                      initial_learning_rate=0.000001, k=None, regr_model='constant',select_function=[4,3,3],min_iter=200,initial_theta = initial_theta,initial_k =1)
    #
    e = EachEpoch()
    e.action_max = np.array([10, 10, 10, 10, 10, 70])
    e.action_min = np.array([-10, -10, -10, -10, -10, 0])

    #jam1
    best_action = bayesian(e,[],iter = 200)

    #jam2
    # best_action = bayesian(e,best_action)

    #jam3
    # best_action = bayesian(e,best_action)




