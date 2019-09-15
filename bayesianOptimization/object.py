import numpy as np
import pygame
from parameter import *
import copy


class Missile(pygame.sprite.Sprite):
    def __init__(self, Location, v):  # 初始化构造函数
        # 初始化图像与位置
        pygame.sprite.Sprite.__init__(self)
        self.image = image_dd
        self.image_o = image_dd
        self.image.set_colorkey(BLACK)
        self.rect = self.image.get_rect()
        self.reset(Location, v)

    def projectile_path(self, dt):  # dd轨迹函数, 暂时设定dd轨迹为抛物线,dd推力抵消了阻力, 暂未设置变轨
        if self.z >= 0:
            # print("DD在Z", self.z)
            self.x += self.speed[0] * dt
            self.y += self.speed[1] * dt
            self.z += (self.speed_pro[2] * dt - 1 / 2 * g * dt ** 2 - g * self.passtime * dt)
        else:
            self.x = self.x
            self.y = self.y
            self.z = self.z

    def shoot(self):  # 发射干扰机动作函数,输出实例化的干扰机类
        if self.jams > 0:  # 只在剩余干扰机数量大于0时执行动作
            jam_type = MAXJAMS - self.jams + 1
            jam = Jam(self.x, self.y, self.z, self.speed, jam_type)
            self.jams -= 1
            return jam
        else:
            return None  # 当干扰即数量小于0,无操作

    def get_missile_pos(self):  # 存储更新的dd位置信息
        return np.array([self.x, self.y, self.z])

    def update(self, dt):  # 更新函数
        if self.z < 0:
            return
        # z方向速度更新
        self.speed[2] = self.speed[2] - g * dt

        # 角度更新
        self.angle = np.degrees(np.arctan2(self.speed[2], self.speed[0])) % 360
        self.image = pygame.transform.rotate(self.image_o, self.angle).convert_alpha()
        self.rect = self.image.get_rect()
        # 轨迹计算

        # 在pygame显示里的距离
        self.rect.centerx = self.x
        self.rect.centery = - self.z + self.y
        self.projectile_path(dt)
        self.passtime += dt

        # 转换成分钟,秒
        self.realtime = np.sqrt(self.passtime * self.passtime * Zd * 0.1)
        ss = self.realtime
        m = int(ss / 60)
        s = int(ss - m * 60)
        self.realtime_mins = str(m) + ':' + str(s)
        # print("passtime",self.passtime)
        # print("dt", dt)

    def change_orbit(self, acc: np.array(3)):
        self.speed += acc

    def reset(self, Location, v):
        self.rect.centerx = Location[0]  # Location为dd初始位置,有三个元素
        self.rect.centery = Location[1]
        # 初始化位置
        self.x = Location[0]
        self.y = Location[1]
        self.z = Location[2]
        # 初始化实例dd初始速度
        self.speed = copy.deepcopy(v)  # V_MISSLE为DD初始速度
        self.speed_pro = copy.deepcopy(v)  # Speed_pro为存储初始速度,不变
        self.passtime = 0
        self.jams = MAXJAMS  # MAXJAMS是dd最大可携带干扰机数量
        self.realtime = 0
        self.realtime_mins = ''
        self.angle = 0


class Jam(pygame.sprite.Sprite):  # 干扰机类
    def __init__(self, x, y, z, speed, type):
        pygame.sprite.Sprite.__init__(self)
        self.image = image_jam
        self.image_o = image_jam
        self.image.set_colorkey(BLACK)
        self.rect = self.image.get_rect()
        self.rect = self.image.get_rect()
        self.rect.centerx = x
        self.rect.centery = y
        self.x = x
        self.y = y
        self.z = z
        self.type = type
        self.speed = copy.deepcopy(speed + DV_JAMS[self.type])
        self.speed_pro = copy.deepcopy(self.speed)
        self.direction = VECTOR_JAMS[type]
        self.passtime = 0
        self.angle = JAM_WORK_ANGLE

    def jam_path(self, dt):
        if self.z >= 0:
            # print("干扰机z方向", self.z)
            self.x += self.speed[0] * dt
            self.y += self.speed[1] * dt
            self.z += self.speed_pro[2] * dt - 1 / 2 * g * dt ** 2 - g * self.passtime * dt
        else:
            self.x = self.x
            self.y = self.y
            self.z = self.z

    def get_direction(self):
        return self.direction

    def update(self, dt):
        self.rect.centerx = self.x
        self.rect.centery = - self.z + self.y
        self.jam_path(dt)
        self.passtime += dt


class Radar(pygame.sprite.Sprite):  # 雷达类
    def __init__(self, L_radar, t, ID):  # self后分别是雷达x，y，z位置坐标及雷达有效工作半径r
        pygame.sprite.Sprite.__init__(self)
        self.image = image_radar
        self.image_o = image_radar
        self.image.set_colorkey(BLACK)
        self.rect = self.image.get_rect()
        self.rect.centerx = L_radar[0]
        self.rect.centery = L_radar[1]
        self.x = L_radar[0]
        self.y = L_radar[1]
        self.z = L_radar[2]
        self.r_radar = L_radar[3]
        self.ID = ID
        self.angle_jam_radar = {}
        self.detection_dd_jam = {}

    def detection_radar_dd_jam(self, jam_sprite, dd_loc):
        # 1.首先判断雷达是否在干扰机1照射范围内
        # 进行角度计算，计算干扰机干扰方向向量与雷达位置方向向量夹角
        for jam in jam_sprite:
            Vector1 = jam.direction
            Vector2 = np.array([jam.x - self.x, jam.y - self.y,
                                jam.z - self.z])  # location_object_jam里的具体元素待定义
            Lv1 = np.sqrt(Vector1.dot(Vector1))
            Lv2 = np.sqrt(Vector2.dot(Vector2))
            cos_angle_jam_radar = Vector1.dot(Vector2) / (Lv1 * Lv2)
            angle_radin_jam_radar = np.arccos(cos_angle_jam_radar)  # 弧度制
            self.angle_jam_radar[jam.type] = math.degrees(angle_radin_jam_radar)  # 角度制

            # print("干扰机", jam.type, "与雷达", self.ID, "角度", self.angle_jam_radar[jam.type])  # debug测试

            # 2.角度计算，判断干扰机与DD相对于雷达的夹角
            Vector3 = np.array([dd_loc[0] - self.x, dd_loc[1] - self.y, dd_loc[2] - self.z])
            Lv3 = np.sqrt(Vector3.dot(Vector3))
            cos_angle_dd_jam = Vector3.dot(Vector2) / (Lv3 * Lv2)
            angle_radin_dd_jam = np.arccos(cos_angle_dd_jam)
            self.angle_dd_jam = math.degrees(angle_radin_dd_jam)

            # 3.计算干扰机1是否成功干扰
            if self.angle_dd_jam < RADAR_DISTINGUISH_ANGLE:  # 条件1,干扰机与DD夹角小于导弹可分辨最小角度
                for key, value in self.angle_jam_radar.items():
                    if value < JAM_WORK_ANGLE:  # 条件2，雷达与干扰机角度小于干扰机干扰工作角度 （意味着雷达被干扰机干扰区域覆盖）
                        self.detection_dd_jam[jam.type] = True  # 干扰机1对于该雷达成功干扰
                    else:
                        self.detection_dd_jam[jam.type] = False
            else:
                self.detection_dd_jam[jam.type] = False  # 干扰机1未对该雷达成功干扰

            # print("干扰机", jam.type, "对于雷达", self.ID, "干扰状态", self.detection_dd_jam[jam.type])  # debug测试

        return self.detection_dd_jam

    def getradarpos(self):
        return np.array([self.x, self.y, self.z])

    def update(self, dt):
        pass

    def reset(self, L_radar):
        self.rect.centerx = L_radar[0]
        self.rect.centery = L_radar[1]
        self.x = L_radar[0]
        self.y = L_radar[1]
        self.z = L_radar[2]
        self.r_radar = L_radar[3]

# class Vector():
#     def __init__(self,x,y,z):
#         self.x = x
#         self.y = y
#         self.z = z
#     def __add__(self,other):
#         x = self.x+other.x
#         y = self.y+other.x
#         z = self.z+other.x
#         return Vector(x,y,z)
#     def __sub__(self, other):
#         x=self.x-other.x
#         y=self.y-other.y
#         z=self.z-other.z
#         return Vector(x,y,z)
#
#     def getvector(self):
#         return (self.x, self.y, self.z)
