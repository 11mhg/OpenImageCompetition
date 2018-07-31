import os, subprocess, time, signal
from data.bbox import Box
import gym
import numpy as np
from PIL import Image
from gym import error, spaces
from gym import utils
import random
import math
from gym.utils import seeding


import logging
logger = logging.getLogger(__name__)

from enum import Enum
class ENVenum(Enum):
    CROPPED=0,
    MOVING=1,
    START=2,
    END=4


class ClassEnv(gym.Env, utils.EzPickle):
    metadata = {'render.modes': ['human']}

    def __init__(self,image=None,gt_boxes=None,anchor_boxes = None,image_size=(800,800)):
        self.viewer = None
        self.image = image
        self.gt_boxes = gt_boxes
        self.image_size = image_size
        self.pred_boxes = Box(cx = random.randint(0,self.image_size[0]),\
            cy=random.randint(0,self.image_size[1]),cw = random.randint(0,10),ch=random.randint(0,10),label=0)
        self.anchor_boxes = anchor_boxes
        self.num_anchors = len(self.anchor_boxes)
        self.final_max_steps = 200
        self.max_steps = 40
        self.prev_iou = 1.0
        #current box
        self.observation_space = spaces.Box(low=np.array([0,0,0,0]), high=np.array([800,800,800,800]),
                                            dtype=np.float32)
        self.found_boxes = []
        self.p_b = random.randint(0,self.num_anchors)
        # Action space omits the Tackle/Catch actions, which are useful on defense
        #Trigger, x, y, w, h
        self.action_space = spaces.Tuple((spaces.Discrete(2),
            spaces.Discrete(self.num_anchors),
            spaces.Box(low=np.array([-self.image_size[0],-self.image_size[1],0,0]),high=np.array([self.image_size[0],self.image_size[1],2,2]),\
                       dtype=np.float32)))
        self.status = ENVenum.START

    def __del__(self):
        return 0
        

    def _configure_environment(self,image=None,gt_boxes=None,image_size=(800,800)):
        """
        Overwritten with new image environment
        """
        self.image_size = image_size
        self.action_space = spaces.Tuple((spaces.Discrete(2),
            spaces.Discrete(self.num_anchors),
            spaces.Box(low=np.array([-self.image_size[0],-self.image_size[1],0,0]),high=np.array([self.image_size[0],self.image_size[1],2,2]),\
                       dtype=np.float32)))
        self.image = image
        self.gt_boxes = gt_boxes
        self.p_b = random.randint(0,self.num_anchors)
        

#    def _start_viewer(self):
#        """
#        Starts the SoccerWindow visualizer. Note the viewer may also be
#        used with a *.rcg logfile to replay a game. See details at
#        https://github.com/LARG/HFO/blob/master/doc/manual.pdf.
#        """
#        cmd = hfo_py.get_viewer_path() +\
#              " --connect --port %d" % (self.server_port)
#        self.viewer = subprocess.Popen(cmd.split(' '), shell=False)


    def _step(self, action):
        self.max_steps -= 1
        self.final_max_steps-=1
        self._take_action(action)
        self.status = ACTION_LOOKUP[action[0]]
        reward = self._get_reward()
        ob = self.pred_box
        if ACTION_LOOKUP[action[0]] == ENVenum.CROPPING or self.max_steps <= 0:   
            iou = self.pred_box.compute_IoU(self.gt_box)
            if iou > 0.6:
                reward = 3
            else:
                reward = -3 
            self.previous_iou = 1.0
            #remove box
            self.pred_boxes = Box(cx = random.randint(0,self.image_size[0]),\
            cy=random.randint(0,self.image_size[1]),cw = random.randint(0,10),ch=random.randint(0,10),label=0)

            self.gt_boxes.remove(self.gt_box)
            self.max_steps = 40
            
        episode_over = self.final_max_steps <=0
        return ob, reward, episode_over, {}
        
    def _take_action(self, action):
        """ Converts the action space into an HFO action. """
        if ACTION_LOOKUP[action[0]] == ENVenum.MOVING:
            self.p_b, d_x, d_y, d_w, d_h = action[1:]
            #write these
            self.pred_box = self.to_box(d_x,d_y,d_w,d_h)
            self.gt_box = self.find_closest() 
            
    def find_closest(self):
        best_ind = 0
        best_iou = 0
        for ind, box in enumerate(self.gt_boxes):
            iou = box.calculate_IoU(self.pred_box)
            if iou > best_iou:
                best_iou = iou
                best_ind = ind
        return self.gt_boxes[best_ind]
            
    def to_box(self, dx, dy, dw, dh):
        p = self.anchor_boxes[self.p_b]
        cx = self.pred_box.cx + dx
        cy = self.pred_box.cy + dy
        cw = p[0]*np.e**(dw)
        ch = p[1]*np.e**(dh)
        assert 0 <= cx <= 800
        assert 0 <= cy <= 800
        return Box(cx=cx,cy=cy,cw=cw,ch=ch)
        

    def _get_reward(self):
        """ Reward is given for scoring a goal. """
        cur_iou = self.pred_box.calculate_IoU(self.gt_box)
        delta_iou = self.previous_iou - cur_iou
        self.previous_iou = cur_iou
        return delta_iou

    def _reset(self):
        self.pred_boxes = Box(cx = random.randint(0,self.image_size[0]),\
            cy=random.randint(0,self.image_size[1]),cw = random.randint(0,10),ch=random.randint(0,10),label=0)
        return self.pred_boxes

#    def _render(self, mode='human', close=False):
#        """ Viewer only supports human mode currently. """
#        if close:
#            if self.viewer is not None:
#                os.kill(self.viewer.pid, signal.SIGKILL)
#        else:
#            if self.viewer is None:
#                self._start_viewer()

def sigmoid(x):
    return 1/(1+math.exp(-x))

ACTION_LOOKUP = {
    0 : ENVenum.MOVING,
    1 : ENVenum.CROPPED # Used only by goalie to catch the ball
}
