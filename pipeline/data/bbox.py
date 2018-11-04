import numpy as np

class Box():

    def __init__(self, x0=None, y0=None, x1=None,y1=None, cx=None, cy=None, w = None, h = None,label=None):
        if x0==None and y0==None and x1==None and y1==None and cx==None and cy==None and w==None and h==None:
            raise ValueError("You must actually input values into either x0y0x1y1 or cxcywh")
        elif x0 == None and y0== None and x1==None and y1==None:
            self.cx = cx
            self.cy = cy
            self.w = w
            self.h = h
            self.cxcy = [self.cx,self.cy,self.w,self.h]
            self.calculate_xyxy()
        elif cx ==None and cy==None and w == None and h == None:
            self.x0=x0
            self.x1=x1
            self.y0=y0
            self.y1=y1
            self.xyxy = [self.x0,self.y0,self.x1,self.y1]
            self.calculate_cxcy()
        else:
            self.x0=x0
            self.x1=x1
            self.y0=y0
            self.y1=y1
            self.xyxy = [self.x0,self.y0,self.x1,self.y1]       
            self.cx = cx
            self.cy = cy
            self.w = w
            self.h = h
            self.cxcy = [self.cx,self.cy,self.w,self.h]
        self.label = label
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError("x1,y1 may be smaller than x0,y0 implying you aren't following standards of top left bottom right (x0y0x1y1). Double Check")

    def calculate_xyxy(self):
        self.x0 = self.cx - (self.w/2.)
        self.y0 = self.cy - (self.h/2.)
        self.x1 = self.cx + (self.w/2.)
        self.y1 = self.cy + (self.h/2.)

    def calculate_cxcy(self):
        self.w = self.x1 - self.x0
        self.h = self.y1 - self.y0
        self.cx = self.x0 + (self.w/2.)
        self.cy = self.y0 + (self.h/2.)

    def compute_IoU(self, box_b):
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

    def area(self):
        return self.w*self.h
