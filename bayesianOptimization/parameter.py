import pygame
import numpy as np
import math
from array import *
from os import path

# todo 甲方要求参数
# DD速度：6-7 m/s
# g：7 m/s^2
# 干扰机delta v >= 5 m/s
# 拦截地点：700 km
# 调姿次数：6 次
# 拦截命中：1 m

# pygame参数,算子图片等
HEIGHT = 849
WIDTH = 1542
DD_WIDTH = 50
DD_HEIGHT = 15
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('DDTF')
img_dir = path.join(path.dirname(__file__), "assets")
background = pygame.image.load(path.join(img_dir, "map.png")).convert_alpha()  # 背景图
background = pygame.transform.scale(background, (WIDTH, HEIGHT))
image_dd = pygame.image.load(path.join(img_dir, "dd.png")).convert_alpha()
image_dd = pygame.transform.scale(image_dd, (DD_WIDTH, DD_HEIGHT))
image_radar = pygame.image.load(path.join(img_dir, "radar.png")).convert_alpha()
image_base = pygame.image.load(path.join(img_dir, "base.png")).convert_alpha()
image_base_red = pygame.image.load(path.join(img_dir, "base_red.png")).convert_alpha()
image_base_blue = pygame.image.load(path.join(img_dir, "base_blue.png")).convert_alpha()
image_jam = pygame.image.load(path.join(img_dir, "plane.png")).convert_alpha()
image_jam = pygame.transform.scale(image_jam, (30, 30))
image_blast = pygame.transform.scale(pygame.image.load(path.join('.', 'assets', 'blast.png')), (60, 80))
image_blast.set_colorkey((2, 2, 2))
# color
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
FONT_SIZE = 22

# 平台类的相关参数


# 后台物理计算参数
JAM_WORK_ANGLE = 60  # 干扰机工作角度
RADAR_DISTINGUISH_ANGLE = 1  # 雷达能区分的角度，应该是0.1°
# 对应到游戏的参数
D = 1000  # 1000像素

# 重力加速度
g = 0.7
# 像素
H = 150
# 变量定义
BASEX = 200
BASEY = HEIGHT - 450
ENEMYBASEX = BASEX + D
ENEMYBASEY = HEIGHT - 450

DISTANCE = 10000000
Zd = DISTANCE / (ENEMYBASEX - BASEX)
LOCK_FPS = 10
# print(1/Zd)
MAXJAMS = 3

MISSILE_LOC = [BASEX, BASEY, 0]
# Radar1_2_Detection_Range = 200
# Radar3_Detection_Range = 400
#
# L_radar1_R = [500, HEIGHT - 500, 8, Radar1_2_Detection_Range]  # 雷达1的参数,最后一项是雷达探测距离
# L_radar2_R = [400, HEIGHT - 400, 8, Radar1_2_Detection_Range]  # 雷达2的参数,最后一项是雷达探测距离
# L_radar3_R = [800, HEIGHT - 550, 8, Radar3_Detection_Range]  # 雷达2的参数,最后一项是雷达探测距离

Radar_s_Detection_Range = 200
Radar_l_Detection_Range = 400

L_radars = [[360, 525, 8, 0],             # 台湾
            [406, 432, 8, Radar_s_Detection_Range],             # 韩国
            [460, 440, 8, Radar_s_Detection_Range],             # 日本南
            [494, 398, 8, Radar_s_Detection_Range],             # 日本北
            [780, 284, 8, Radar_l_Detection_Range],             # 美国海1
            [790, 284, 8, Radar_s_Detection_Range],             # 美国海2
            [896, 168, 8, 0],             # 美国阿
            [1148, 392, 8, 0],            # 美国西
            [1458, 406, 8, 0],            # 美国东
            ]

T = np.ceil(2 * (np.sqrt(2 * g * H) / g))  # 总时间设为二倍的 t=Vz/g
# print([D / T, 0, np.sqrt(2 * g * H)])

# DD速度变量
# [28.571428571428573, 0, 17.146428199482248]
V_MISSILE = np.array([D / T, 0, np.sqrt(2 * g * H)])  # vz = 根号下2g*H
# print(V_MISSILE)
V_MISSILE_pro = np.array([D / T, 0, np.sqrt(2 * g * H)])
V_MISSILE_C = np.array([0.1, 0.1, 0.1])   # 变轨机动性能参数

# 干扰机动作变量
DV_JAMS = {1: [0, -5.3, -6.2], 2: [-5, 4.3, -4], 3: [-3, -4.3, -2]}
VECTOR_JAMS = {1: np.array([1, 1, 1]), 2: np.array([2, 1, 3]), 3: np.array([1, 1, 1])}

# redbase_loc = (200, 400)
# bluebase_loc = (1200, 400)

##################
c_orbit_acc = 1

def calculate_angle(Vector1, Vector2):
    # location_object_jam里的具体元素待定义
    Lv1 = np.sqrt(Vector1.dot(Vector1))
    Lv2 = np.sqrt(Vector2.dot(Vector2))
    cos_angle = Vector1.dot(Vector2) / (Lv1 * Lv2)
    angle = np.arccos(cos_angle)  # 弧度制
    angle = math.degrees(angle)  # 角度制
    return angle


def distance_dd_radar(missloc, l_radar):
    distance_radar = np.sqrt(np.sum((missloc - l_radar) ** 2))
    return distance_radar
