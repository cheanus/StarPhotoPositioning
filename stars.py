import cv2
import numpy as np

def get_user_input(image):
    # 显示图像并等待用户点击
    cv2.namedWindow('Image', cv2.WINDOW_FREERATIO)
    cv2.imshow('Image', image)
    print("请点击月亮的粗略位置，并按下 'enter' 键确认:")
    roi = cv2.selectROI('Image', image, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    
    # 返回用户选择的区域
    return roi

def find_moon_center(image, roi):
    # 裁剪图像为用户选择的区域
    roi_image = image[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]

    # 在转换为灰度图之前应用高斯模糊
    blurred = cv2.convertScaleAbs(roi_image, alpha=1, beta=3)
    # blurred = cv2.GaussianBlur(blurred, (9, 9), 1000)
    blurred = cv2.GaussianBlur(blurred, (9, 9), 0)
    blurred = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    
    # 霍夫圆检测
    ## 月亮检测
    # circle = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 20,
    #                         param1=30, param2=10, minRadius=40, maxRadius=50)
    ## 星星检测
    circle = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 20,
                            param1=10, param2=5, minRadius=0, maxRadius=5)
    center = []
    # 将检测结果绘制在图像上
    for i in circle[0, :]:  # 遍历矩阵的每一行的数据
        # 绘制圆形
        cv2.circle(roi_image, (int(i[0]), int(i[1])), int(i[2]), (255, 0, 0), 1)
        # 绘制圆心
        cv2.circle(roi_image, (int(i[0]), int(i[1])), 2, (0, 255, 0), 1)
        center.append((i[0]+roi[0], i[1]+roi[1]))
    
    if len(center) > 1:
        print('Warning: 找到两个及以上的圆心。')
    # 输出中心点坐标
    print("中心点坐标:", center)

    # 显示标记后的图像
    cv2.namedWindow('Image', cv2.WINDOW_FREERATIO)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return center

def main(image):
    # 获取用户输入
    user_roi = get_user_input(image)

    # 查找月亮中心点
    find_moon_center(image, user_roi)

if __name__ == '__main__':
    # 读取图像
    image_path = 'images/night_sky.jpg'  # 替换为实际图像路径
    image = cv2.imread(image_path)
    main(image)