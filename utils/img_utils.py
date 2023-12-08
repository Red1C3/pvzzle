import cv2
def read_img(path,grayscale=False):
    return cv2.imread(path,cv2.IMREAD_GRAYSCALE if grayscale else None)

def display_img(img,title="title"):
    cv2.namedWindow(title,cv2.WINDOW_NORMAL)
    cv2.imshow(title,img)
    cv2.waitKey(0)

def close_all_windows():
    cv2.destroyAllWindows()

def sobel_vertical_whole_img(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sobel= cv2.Sobel(img,cv2.CV_16S,0,1,ksize=3)
    return cv2.convertScaleAbs(sobel)
