import os
import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
from sklearn.cluster import DBSCAN

def cluster_center(points, eps, min_samples=2):
    # 聚类，寻找数量最多的簇的中心
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    unique_labels = np.unique(labels)
    max_label = -1
    max_count = 0
    for label in unique_labels:
        count = np.sum(labels == label)
        if count > max_count and label != -1:
            max_count = count
            max_label = label
    avg_intersection = np.mean(points[labels == max_label], axis=0)
    return avg_intersection

def plumb_line(img, args):
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
    
    lines = cv2.HoughLinesP(
        result, 1, np.pi / 180,
        args['HoughLinesP']['threshold'],
        minLineLength=args['HoughLinesP']['minLineLength'],
        maxLineGap=args['HoughLinesP']['maxLineGap']
    )
    return lines

def lines_filter(img, lines):
    fix_lines = []
    fix_result_img = img.copy()

    expected_center = args['expected_center']
    radius_threshold = args['expected_radius']
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 == x2:
            theta = np.pi/2
        else:
            theta = np.arctan((y2-y1)/(x2-x1))
        # 计算与预期中心的距离
        distance = np.abs((y2-y1)*expected_center[0]-(x2-x1)*expected_center[1]+x2*y1-y2*x1)/np.sqrt((x2-x1)**2+(y2-y1)**2)
        # 筛选长度和角度
        if args['ignore_angle_scope'][0] < theta < args['ignore_angle_scope'][1] or distance > radius_threshold:
            continue
        cv2.line(fix_result_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        fix_lines.append(line[0])
    # plt.imshow(fix_result[:, :, ::-1])
    return fix_lines, fix_result_img

def find_nearest_point(lines, args):
    def funs(point):
        x, y = point
        value = 0
        for x1, y1, x2, y2 in lines:
           value += np.abs((y1-y2)*x+(x2-x1)*y+x1*y2-x2*y1)/np.sqrt((x2-x1)**2+(y2-y1)**2)
        return value
    x0, y0 = args['expected_center']
    point = fmin(funs, (x0, y0), disp=False)
    return point
    
def main(image_path, out_dir, args):
    img = cv2.imread(image_path)
    lines = plumb_line(img, args)

    while args['expected_radius'] > 1:
        fix_lines, fix_result_img = lines_filter(img, lines)
        if len(fix_lines) <= 1:
            break
        sky_top = find_nearest_point(fix_lines, args)
        args['expected_radius'] = 1.1 * np.sqrt(((sky_top - args['expected_center'])**2).sum())
        args['expected_center'] = sky_top

    plt.imshow(fix_result_img[:, :, ::-1])
    plt.scatter(sky_top[0], sky_top[1], c='r', s=50)
    out_path = os.path.join(out_dir, 'sky_top.jpg')
    plt.savefig(out_path)
    print('天顶点坐标: ', sky_top)

if __name__ == '__main__':
    args = yaml.safe_load(open("config.yaml"))
    args['expected_center'] = np.array(args['expected_center'])
    args['ignore_angle_scope'] = np.array(args['ignore_angle_scope'])/180*np.pi

    image_path = args['img_path']
    out_dir = args['out_dir']
    main(image_path, out_dir, args)