import cv2
def read_img(path,grayscale=False):
    return cv2.imread(path,cv2.IMREAD_GRAYSCALE if grayscale else None)

def display_img(img,title="title"):
    cv2.namedWindow(title,cv2.WINDOW_NORMAL)
    cv2.imshow(title,img)
    cv2.waitKey(0)

def close_all_windows():
    cv2.destroyAllWindows()