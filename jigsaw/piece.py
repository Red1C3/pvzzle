class Piece:
    def __init__(self,x,y,w,h,sub_img,mask,contour,type):
        self.x=x
        self.y=y
        self.w=w
        self.h=h
        self.sub_img=sub_img
        self.mask=mask
        self.contour=contour
        self.type=type