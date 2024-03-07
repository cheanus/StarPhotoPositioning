import os
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def cluster_center(points, eps=100, min_samples=2):
    # 聚类，寻找数量最多的簇的中心
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    unique_labels = np.unique(labels)
    max_label = -1
    max_count = 0
    for label in unique_labels:
        count = np.sum(labels == label)
        if count > max_count:
            max_count = count
            max_label = label
    avg_intersection = np.mean(points[labels == max_label], axis=0)
    return avg_intersection

def plumb_line(img):
    blurred = cv2.convertScaleAbs(img, alpha=2, beta=6)
    # 将彩色图片灰度化
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    # plt.imshow(gray, cmap='gray')

    # 创建一个LSD对象
    lsd = cv2.createLineSegmentDetector(0)
    # 执行检测结果
    dlines = lsd.detect(gray)
    result = np.zeros_like(gray)

    # 绘制检测结果
    for dline in dlines[0]:
        x0 = int(round(dline[0][0]))
        y0 = int(round(dline[0][1]))
        x1 = int(round(dline[0][2]))
        y1 = int(round(dline[0][3]))
        cv2.line(result, (x0, y0), (x1,y1), 255, 5, cv2.LINE_AA)
    # plt.imshow(result, cmap='gray')
    
    lines = cv2.HoughLinesP(result, 1, np.pi / 180, 40, minLineLength=100, maxLineGap=30)
    fix_lines = []
    fix_result = img.copy()

    expected_center = (1600, -4000)
    radius_threshold = 400
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 == x2:
            theta = np.pi/2
        else:
            theta = np.arctan((y2-y1)/(x2-x1))
        # 计算与预期中心的距离
        distance = np.abs((y2-y1)*expected_center[0]-(x2-x1)*expected_center[1]+x2*y1-y2*x1)/np.sqrt((x2-x1)**2+(y2-y1)**2)
        # 筛选长度和角度
        if -np.pi/4 < theta < np.pi/4 or distance > radius_threshold:
            continue
        cv2.line(fix_result, (x1, y1), (x2, y2), (0, 0, 255), 2)
        fix_lines.append(line[0])
    # plt.imshow(fix_result[:, :, ::-1])
    return fix_lines
        
def find_avg_intersection(lines):
    intersections = []
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            x0, y0, x1, y1 = lines[i]
            x2, y2, x3, y3 = lines[j]
            if (x1-x0)*(y3-y2) == (x3-x2)*(y1-y0):
                continue
            x = ((x1*y0-x0*y1)*(x3-x2)-(x1-x0)*(x3*y2-x2*y3))/((x1-x0)*(y3-y2)-(x3-x2)*(y1-y0))
            y = ((x1*y0-x0*y1)*(y3-y2)-(y1-y0)*(x3*y2-x2*y3))/((x1-x0)*(y3-y2)-(x3-x2)*(y1-y0))
            intersections.append((x, y))
    intersections = np.array(intersections)
    avg_intersection = cluster_center(intersections)
    return avg_intersection

def main(image_path, out_dir):
    img = cv2.imread(image_path)
    fix_lines = plumb_line(img)
    sky_top = find_avg_intersection(fix_lines)
    sky_top_img = img.copy()
    plt.imshow(sky_top_img[:, :, ::-1])
    plt.scatter(sky_top[0], sky_top[1], c='r', s=50)
    out_path = os.path.join(out_dir, 'sky_top.jpg')
    plt.savefig(out_path)
    print('天顶点坐标: ', sky_top)

if __name__ == '__main__':
    image_path = 'images/night_sky.jpg'  # 替换为实际图像路径
    out_dir = 'output'
    main(image_path, out_dir)