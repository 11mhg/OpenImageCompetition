import numpy as np

class Box():

    def calculate_cxcy(self,x0=0,y0=0,x1=0,y1=0):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        
        
        self.w = (self.x1-self.x0)
        self.h = (self.y1-self.y0)
        self.cx = self.x0+(self.w/2.)
        self.cy = self.y0+(self.h/2.)
        self.cxcy = [self.cx,self.cy,self.w,self.h]

    def calculate_xyxy(self,cx=0,cy=0,w=0,h=0):
        self.w = w
        self.h = h
        self.cx = cx
        self.cy = cy

        self.x0 = self.cx-(self.w/2.)
        self.x1 = self.cx+(self.w/2.)
        self.y0 = self.cy-(self.h/2.)
        self.y1 = self.cy+(self.h/2.)
        self.cxcy = [self.cx,self.cy,self.w,self.h]
        self.xyxy = [self.x0,self.y0,self.x1,self.y1]


    def __init__(self, x0=0, y0=0, x1=0, y1=0,cx = 0, cy = 0, cw = 0, ch = 0,label=0):
        if x1 < x0 or y1 < y0:
            raise ValueError("x1, y1, may be larger than x0, y0 implying that you are not using the standardized top left, bottom right. Double check")
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.cx = cx
        self.cy = cy
        self.ch = ch
        self.cw= cw
        self.label = 0
        self.cxcy = [self.cx,self.cy,self.cw,self.ch]
        self.xyxy = [self.x0,self.y0,self.x1,self.y1]
    
    def area(self):
        return self.w*self.h

    def compute_IoU(self,box_b):
        a = self.xyxy
        b = box_b.xyxy
        max_xy = np.minimum(a[2:],b[2:])
        min_xy = np.maximum(a[:2],b[:2])
        diff_xy = max_xy - min_xy
        inter = diff_xy[0] * diff_xy[1]
        if inter <=0.0:
            return 0.0
        union = a.area() + b.area() - inter
        return inter/union
        
