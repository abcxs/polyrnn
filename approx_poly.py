import cv2
import json
import math
import numpy as np
import os
import SimpleITK as sitk
from PIL import Image


def get_p2l_dis(point_x: int, point_y: int, line_x1: int, line_y1: int, line_x2: int, line_y2: int) -> float:
    """
    【计算点P到直线L的最短距离】
    :param point_x: P的X坐标，int（在本工程中出现的所有坐标均采用整数型）
    :param point_y: P的Y坐标
    :param line_x1: 直线上的点L1的X坐标
    :param line_y1: 直线上的点L1的Y坐标
    :param line_x2: 直线上的点L2的X坐标
    :param line_y2: 直线上的点L2的Y坐标
    :return: 距离值，float
    """
    va = line_y2 - line_y1
    vb = line_x1 - line_x2
    vc = line_x2 * line_y1 - line_x1 * line_y2
    dis = (math.fabs(va * point_x + vb * point_y + vc)) / (math.pow(va * va + vb * vb, 0.5))
    return dis


def polygon_fitting_recursive(points, pbegin: int, pend: int, delta: float):
    """
    【多边形轮廓拟合-递归子函数】
    该函数会将点集中需要删除的点全部修改为(-1, -1)，而不真的去删除那些点
    :param points: 轮廓点集，n*1*2的ndarray
    :param pbegin: 当前考虑的曲线段中的起点下标，int
    :param pend: 当前考虑的曲线段中的终点下标（含），int
    :param delta: 拟合阈值，继承自主函数，float
    :return: none
    """
    max_dis = -1  # 最远距离
    max_p = -1  # 最远距离的点下标
    p1 = points[pbegin][0]  # 曲线段的起点，长度为2的ndarray
    p3 = points[pend][0]  # 曲线段的终点
    if pend == 0:
        px_range = range(pbegin + 1, len(points))  # px_range是当前考虑的曲线段上的点的取值范围
    else:
        px_range = range(pbegin + 1, pend)
    for px in px_range:
        p2 = points[px][0]  # p2是当前考虑的点
        dis = get_p2l_dis(p2[0], p2[1], p1[0], p1[1], p3[0], p3[1])  # 计算p2到直线p1-p2的最短距离
        if dis > max_dis:
            max_dis = dis
            max_p = px  # 找到距离直线p1-p2最远的那个点
    if max_dis == -1 or max_p == -1:
        return
    if delta >= max_dis >= 0:  # 最远距离小于拟合阈值delta
        for px in px_range:  # 将p1-p2内的全部点删除（暂时标记为-1，不实际删除）
            points[px][0][0] = -1
            points[px][0][1] = -1
    else:  # 存在最远距离大于拟合阈值delta，则进行二分
        polygon_fitting_recursive(points, pbegin, max_p, delta)
        polygon_fitting_recursive(points, max_p, pend, delta)


def polygon_fitting_main(points, delta: float, flag_debug=False):
    """
    【多边形轮廓拟合-主函数】
    :param points: 轮廓点集，其中多余的点会在原集合上被直接删除，n*1*2的ndarray
    :param delta:拟合阈值，float
    :param flag_debug: 是否输出调试信息
    :return: 拟合后的点集
    """
    if len(points) < 3:
        # print('ERROR: the length of points is too small!', len(points))
        return points
    pbegin = 0  # 起点
    pmid = len(points) // 2  # 中点
    pend = pbegin  # 终点（和起点相同）
    polygon_fitting_recursive(points, pbegin, pmid, delta)  # 将整个轮廓线一分为二进行拟合
    polygon_fitting_recursive(points, pmid, pend, delta)
    p_i = 0
    while p_i < len(points):
        if points[p_i][0][0] == -1:  # 删除所有被递归子函数标记为(-1, -1)的点
            if flag_debug:
                print('[DEBUG polygon_fitting_main] delete point = ', points[p_i])
            points = np.delete(points, p_i, axis=0)
            continue
        p_i = p_i + 1
    return points


def polygon_fitting(points, delta: float, image=None, times=10, flag_image=True, flag_text=True, flag_debug=False,
                    color_point=(0, 0, 255), color_line=(0, 255, 0), color_text=(0, 127, 255)):
    """
    【多边形轮廓拟合-调用函数】
    :param points: 轮廓点集，其中多余的点会在原集合上被直接删除，n*1*2的ndarray
    :param delta:拟合阈值，float
    :param image: 展示图像（不修改原图像）
    :param times: 轮切次数，int
    :param flag_image: 处理后是否在image上展示图像
    :param flag_text: 处理后是否在image上显示剩余点的个数
    :param flag_debug: 是否输出调试信息
    :param color_point: 画图时的点的颜色
    :param color_line: 画图时的线的颜色
    :param color_text: 画图时的字的颜色
    :return: 拟合后的点集
    """
    if times <= 0:
        print('ERROR: times must > 0! ', times)
    before_num = len(points)  # 处理前的点集的大小
    for time in range(times):  # 循环初拟合
        points = polygon_fitting_main(points, delta, flag_debug)  # 多边形初拟合
        p_first = np.copy(points[0])  # 轮切点集
        points = np.delete(points, 0, axis=0)
        points = np.insert(points, len(points), p_first, axis=0)
    # print('polygon_fitting: ', before_num, ' -> ', len(points))  # 处理后的点集的大小
    if flag_image:
        if image is None:
            print('WARNING: polygon_fitting image is none, cant display this image!')
            return points
        img = image.copy()
        if flag_text:
            num = str(len(points))
            text_x = int(max(points[:, 0, 0]) + min(points[:, 0, 0])) // 2 - 5
            text_y = int(max(points[:, 0, 1]) + min(points[:, 0, 1])) // 2 + 5
            cv2.putText(img, num, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX, 1, color=color_text)
        cv2.drawContours(img, [points], -1, color_line, 0)
        cv2.drawContours(img, points, -1, color_point, 3)
        cv2.namedWindow('polygon_fitting', 0)
        cv2.resizeWindow("polygon_fitting", 1024, 1024)
        cv2.imshow("polygon_fitting", img)
        cv2.waitKey()
    return points


def add_line_points(p1, p2, max_num: int = 25, flag_num=True, flag_debug=False):
    """
    【获得两点间的线段点集】
    在线段p1-p2上不断补点，最终返回一个线段的点集
    :param p1: 起点，1*2的ndaray
    :param p2: 终点，1*2的ndarray
    :param max_num: 最多的点的数量，int
    :param flag_num: 是否限制点的数量
    :param flag_debug: 是否输出调试信息
    :return: 起点为p1，终点为p2的线段点集，n*1*2的ndarray
    """
    if p1[0][0] > p2[0][0]:
        temp_p = p1
        p1 = p2
        p2 = temp_p
    k = (p2[0][1] - p1[0][1] + 0.000001) / (p2[0][0] - p1[0][0] + 0.000001)  # 直线斜率k = (y2-y1) / (x2-x1)
    b = p1[0][1] - k * p1[0][0]  # 直线偏置b = y - kx
    ans = np.array([[p1[0][0], p1[0][1]], [p2[0][0], p2[0][1]]])  # 2*2 的ndarray
    if k > 1 or k < -1:  # y的跨度大
        for y_i in range(min(p1[0][1], p2[0][1]) + 1, max(p1[0][1], p2[0][1])):
            ans = np.insert(ans, -2, [int((y_i - b + 0.000001) / k), y_i])  # 在倒数第二个元素前插入2个元素（因为ans一行有2个元素）
    else:  # x的跨度大
        for x_i in range(min(p1[0][0], p2[0][0]) + 1, max(p1[0][0], p2[0][0])):
            ans = np.insert(ans, -2, [x_i, int(k * x_i + b)])
    ans = ans.reshape((-1, 1, 2))  # 重塑成n*1*2的ndarray
    if flag_num and (len(ans) > max_num):  # 如果点的数量超过阈值，则削减点的数量
        width = (len(ans) - 2) // (max_num - 2)  # 宽度
        few_ans = np.array([ans[0], ans[-1]])
        for time in range(max_num - 2):
            few_ans = np.insert(few_ans, -1, ans[width * (time + 1)], axis=0)
        return few_ans
    if flag_debug:
        print('[DEBUG add_line_points] p1=', p1[0], ', p2=', p2[0], ', k=', k, ', b=', b)
    return ans


def cal_hausdorff_distance(points1, points2):
    """
    【计算单向Hausdorff距离】
    :param points1: 轮廓点集1，n*1*2的ndarray
    :param points2: 轮廓点集2，n*1*2的ndarray
    :return: 点集1到点集2的单向Hausdorff距离
    """
    min_list = []
    for pa in points1:  # 从点集points1中取一个点pa
        min_dis = 999999999
        min_pb = -1
        for pb in points2:  # 计算点pa到点集points2的最短距离
            dis = np.linalg.norm(pa - pb)  # 计算两点间的欧氏距离
            if dis < min_dis:
                min_dis = dis
                min_pb = pb
        min_list.append(min_dis)
    return max(min_list)  # 单向Hausdorff距离就是所有最短距离的最大值


def get_min_area_rect(points):
    """
    【得到点集的最小面积外接矩形】
    :param points: 轮廓点集，n*1*2的ndarray
    :return: 最小面积外接矩形的四个端点，4*1*2的ndarray
    """
    rect = cv2.minAreaRect(points)  # 最小面积外接矩形
    box = cv2.boxPoints(rect)  # 得到矩形的四个端点
    box = np.int0(box)
    box = box[:, np.newaxis, :]  # 从4*2转化为4*1*2
    return box


def hausdorff_fitting(points, delta_r: float, image=None,  box=None, flag_image=True, flag_debug=False, flag_debug2=False, flag_turn=False,
                      color_box=(255, 0, 0), color_point=(0, 0, 255), color_line=(0, 255, 0), color_old=(0, 127, 255), color_new=(0, 255, 255)):
    """
    【Hausdorff深度规整】
    首先获得points的最小面积外接矩形，然后根据Hausdorff距离，将mask上的点往外接矩形上偏移
    :param points: 轮廓点集，n*1*2的ndarray
    :param delta_r: 规整阈值，float
    :param image: 展示图像（不修改原图像）
    :param box: 最小边界外接矩形的四个端点，4*1*2的ndarray
    :param flag_image: 处理后是否在image上展示图像
    :param flag_debug: 是否输出调试信息
    :param flag_debug2: 是否输出调试信息2
    :param flag_turn: 是否逐个输出规整示意图
    :param color_box: 画图时的最小面积外接矩形的颜色
    :param color_point: 画图时的点的颜色
    :param color_line: 画图时的线的颜色
    :param color_old: 规整示意图中旧的点的颜色
    :param color_new: 规整示意图中规整后的点的颜色
    :return: 规整后的轮廓点集，n*1*2的ndarray
    """
    if box is None:
        box = get_min_area_rect(points)  # 得到最小边界外接矩形的四个顶点，4*1*2的ndarray
    box_line1 = add_line_points(box[0], box[1], flag_debug=flag_debug)  # 得到外接矩形边界上的所有点
    box_line2 = add_line_points(box[1], box[2], flag_debug=flag_debug)
    box_line3 = add_line_points(box[2], box[3], flag_debug=flag_debug)
    box_line4 = add_line_points(box[3], box[0], flag_debug=flag_debug)
    box_line = np.vstack((box_line1, box_line2, box_line3, box_line4))  # 堆叠，形成n*1*4的点集
    s_build = cv2.contourArea(points)  # 得到mask的面积
    s_rect = cv2.contourArea(box)  # 得到最小边界外接矩形的面积
    l_min = math.sqrt(min((box[0][0][0]-box[1][0][0])**2 + (box[0][0][1]-box[1][0][1])**2,
                          (box[1][0][0]-box[2][0][0])**2 + (box[1][0][1]-box[2][0][1])**2))  # 得到外接矩形短边长度
    delta = (s_build / s_rect) * delta_r * (l_min / 2)  # 动态计算阈值
    if flag_debug:
        print('[DEBUG hausdorff_fitting] s_build = %f, s_rect=%f, l_min=%f, delta=%f,' % (s_build, s_rect, l_min, delta), end=' ')
        print('box=', box[0][0], box[1][0], box[2][0], box[3][0])
    turn_count = 0  # 规整计数器
    for p_i in range(len(points)):
        p_j = (p_i + 1) % len(points)
        pi = points[p_i]  # 起点
        pj = points[p_j]  # 终点
        house_line = add_line_points(pi, pj, flag_num=True)  # 相邻两点间的线段
        hausdorff_dis = cal_hausdorff_distance(house_line, box_line)  # 求单向Hausdorff距离
        if hausdorff_dis < delta:  # 小于阈值，可以进行规整
            p = [pi, pj]
            turn_count = turn_count + 1  # 计数器+1
            for p_k in range(len(p)):  # 分别取起点和终点
                min_dis = 999999999
                min_l = -1
                for line_i in range(len(box)):  # 计算该点离哪个边界线最近
                    dis = get_p2l_dis(p[p_k][0][0], p[p_k][0][1], box[line_i][0][0], box[line_i][0][1], box[(line_i + 1) % len(box)][0][0], box[(line_i + 1) % len(box)][0][1])
                    if dis < min_dis:
                        min_l = line_i
                        min_dis = dis
                x1 = box[min_l][0][0]  # 取拥有最短距离的边界线的端点p1
                y1 = box[min_l][0][1]
                x2 = box[(min_l + 1) % len(box)][0][0]  # 取拥有最短距离的边界线的端点p2
                y2 = box[(min_l + 1) % len(box)][0][1]
                x0 = p[p_k][0][0]  # 当前取的点p0
                y0 = p[p_k][0][1]
                u = ((x1-x2)*(x0-x1) + (y1-y2)*(y0-y1)) / ((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))
                xp = int(x1 + u * (x1-x2))  # p0在直线p1-p2上的垂足为pp
                yp = int(y1 + u * (y1-y2))
                if flag_debug2:
                    print('[DEBUG hausdorff_fitting_turn] ', end=' ')
                    print(('(%4d, %4d) -> (%4d, %4d)  min_dis=%f, min_l=%d') % (p[p_k][0][0], p[p_k][0][1], xp, yp, min_dis, min_l))
                if flag_turn:
                    if image is None:
                        print('WARNING: hausdorff_fitting_turn image is none, cant display this image!')
                    else:
                        img = image.copy()
                        cv2.drawContours(img, box, -1, color_box, 2)  # 绘制最小面积外接矩形
                        cv2.drawContours(img, [box], -1, color_box, 0)
                        cv2.drawContours(img, [points], -1, color_line, 0)  # 绘制轮廓线
                        cv2.drawContours(img, points, -1, color_point, 3)
                        cv2.circle(img, (p[p_k][0][0], p[p_k][0][1]), 1, color_old, 3)  # 绘制规整示意点
                        cv2.circle(img, (xp, yp), 1, color_new, 3)
                        cv2.namedWindow('hausdorff_fitting_turn', 1)
                        cv2.resizeWindow("hausdorff_fitting_turn", 1024, 1024)
                        cv2.imshow("hausdorff_fitting_turn", img)
                        cv2.waitKey()
                p[p_k][0][0] = xp  # 将点规整
                p[p_k][0][1] = yp
        else:  # 大于等于阈值
            if flag_debug2:
                print('[DEBUG hausdorff_fitting_turn] ', end=' ')
                print(('(%4d, %4d) (%4d, %4d) hausdorff_dis=%f, delta=%f') % (pi[0][0], pi[0][1], pj[0][0], pj[0][1], hausdorff_dis, delta))
            if flag_turn:
                img = image.copy()
                cv2.drawContours(img, box, -1, color_box, 2)  # 绘制最小面积外接矩形
                cv2.drawContours(img, [box], -1, color_box, 0)
                cv2.drawContours(img, [points], -1, color_line, 0)  # 绘制轮廓线
                cv2.drawContours(img, points, -1, color_point, 3)
                cv2.circle(img, (pi[0][0], pi[0][1]), 1, color_old, 3)  # 绘制规整示意点
                cv2.circle(img, (pj[0][0], pj[0][1]), 1, color_new, 3)
                cv2.namedWindow('hausdorff_fitting_turn', 1)
                cv2.resizeWindow("hausdorff_fitting_turn", 1024, 1024)
                cv2.imshow("hausdorff_fitting_turn", img)
                cv2.waitKey()
    # print('hausdorff_fitting: turn_count = %d (%d / %d)' % (turn_count, turn_count, len(points)))
    if flag_image:
        if image is None:
            print('WARNING: hausdorff_fitting image is none, cant display this image!')
        else:
            img = image.copy()
            cv2.drawContours(img, box, -1, color_box, 2)  # 绘制最小面积外接矩形
            cv2.drawContours(img, [box], -1, color_box, 0)
            cv2.drawContours(img, [points], -1, color_line, 0)  # 绘制轮廓线
            cv2.drawContours(img, points, -1, color_point, 3)
            cv2.namedWindow('hausdorff_fitting', 1)
            cv2.resizeWindow("hausdorff_fitting", 1024, 1024)
            cv2.imshow("hausdorff_fitting", img)
            cv2.waitKey()
    return points


def fill_4points(points, image=None, box=None, flag_image=True, flag_debug=False, flag_turn=False, color_box=(255, 0, 0),
                 color_point=(0, 0, 255), color_line=(0, 255, 0), color_old=(0, 127, 255), color_new=(0, 255, 255)):
    """
    【补全四边形的四个顶角】
    :param points: 轮廓点集，n*1*2的ndarray
    :param image: 展示图像（不修改原图像）
    :param box: 最小边界外接矩形的四个端点，4*1*2的ndarray
    :param flag_image: 处理后是否在image上展示图像
    :param flag_debug: 是否输出调试信息
    :param flag_turn: 是否逐个输出规整示意图
    :param color_box: 画图时的最小面积外接矩形的颜色
    :param color_point: 画图时的点的颜色
    :param color_line: 画图时的线的颜色
    :param color_old: 规整示意图中旧的点的颜色
    :param color_new: 规整示意图中规整后的点的颜色
    :return: 补全顶角后的点集，n*1*2的ndarray
    """
    before_num = len(points)  # 处理前的点集的大小
    if box is None:
        box = get_min_area_rect(points)  # 得到最小边界外接矩形的四个顶点，4*1*2的ndarray
    s_build = cv2.contourArea(points)  # 得到mask的面积
    s_rect = cv2.contourArea(box)  # 得到最小边界外接矩形的面积
    l_min = math.sqrt(min((box[0][0][0] - box[1][0][0]) ** 2 + (box[0][0][1] - box[1][0][1]) ** 2,
                          (box[1][0][0] - box[2][0][0]) ** 2 + (box[1][0][1] - box[2][0][1]) ** 2))  # 得到外接矩形短边长度
    l_max = math.sqrt(max((box[0][0][0] - box[1][0][0]) ** 2 + (box[0][0][1] - box[1][0][1]) ** 2,
                          (box[1][0][0] - box[2][0][0]) ** 2 + (box[1][0][1] - box[2][0][1]) ** 2))  # 得到外接矩形长边长度
    k = [0.1, 0.1, 0.1, 0.1]  # 4条边界线的斜率k，k=(y1-y2)/(x1-x2)
    b = [0.1, 0.1, 0.1, 0.1]  # 4条边界线的偏置b
    k[0] = (box[0][0][1] - box[1][0][1] + 0.000001) / (box[0][0][0] - box[1][0][0] + 0.000001)
    k[1] = (box[1][0][1] - box[2][0][1] + 0.000001) / (box[1][0][0] - box[2][0][0] + 0.000001)
    k[2] = k[0]
    k[3] = k[1]
    for l_i in range(len(box)):  # 计算4个偏置，b=y-kx
        b[l_i] = box[l_i][0][1] - k[l_i] * box[l_i][0][0]
    if flag_debug:
        print('[DEBUG fill_4points] global: len(points)=', len(points), ', length=', l_min, l_max, ', box=', box[0][0], box[1][0], box[2][0], box[3][0], ', k=', k, ', b=', b)
    points_on_lines = [[], [], [], []]  # 轮廓上的点位于外接矩形的边上的点集
    points_del = []  # 打算被删除的点
    for p_i in range(len(points)):  # 枚举轮廓上的每个点
        for l_i in range(len(k)):  # 判断该点是否位于哪条外接矩形的边直线上
            if abs((k[l_i] * points[p_i][0][0] + b[l_i]) - points[p_i][0][1]) <= 3 or abs((points[p_i][0][1] - b[l_i] + 0.000001) / (k[l_i] + 0.00001) - points[p_i][0][0]) <= 3:
                if flag_debug:
                    print('[DEBUG fill_4points] points_on_line: point %d (%d, %d) is on line %d : y=%f x + %f' % (p_i, points[p_i][0][0], points[p_i][0][1], l_i, k[l_i], b[l_i]))
                points_on_lines[l_i].append(p_i)  # 该点存在于边界线上，记录其下标
    if flag_debug:
        for l_i in range(len(points_on_lines)):
            print('[DEBUG fill_4points] points_on_lines', l_i, ' :', end=' ')
            for p_i in points_on_lines[l_i]:
                print('%d(%d, %d)' % (p_i, points[p_i][0][0], points[p_i][0][1]), end=' | ')
            print('')
    direction = -999999999  # 遍历方向
    direction_flag = False  # 遍历方向标志位
    for box_point_i in range(len(box)):  # 核心功能，分别补齐最小外接矩形的四个顶角
        min_l_dis = [999999999.999, 999999999.999]  # 距离该顶角最近的两个点（位于不同边）上的最短距离
        min_i = [99999, 99999]
        for box_line_i in range(len(min_l_dis)):  # 在该顶角所在的2条直线上，分别计算距离该顶点最近的2个轮廓点
            for p_i in points_on_lines[(len(box) + box_point_i - box_line_i) % len(box)]:  # 枚举位于该直线上的点
                dis = math.sqrt((box[box_point_i][0][0] - points[p_i][0][0]) ** 2 + (box[box_point_i][0][1] - points[p_i][0][1]) ** 2)
                if dis <= min_l_dis[box_line_i]:
                    min_l_dis[box_line_i] = dis
                    min_i[box_line_i] = p_i
        if min(min_l_dis) <= 0.5:
            # print('continue!!!!!!!!!!!!!!!', min_l_dis, min_i)
            continue
        min_index = -1
        max_index = -1
        if box_point_i == 0 or max(min_i) == 99999 or direction_flag is False:  # 获得遍历方向
            direction = min_i[1] - min_i[0]
        if abs(min(min_l_dis) - l_min) <= l_min / 10:  # 如果某条最短边已经快延伸到相邻顶点时，则将距离置为1
            min_index = min_l_dis.index(min(min_l_dis))
            min_l_dis[min_index] = min_l_dis[min_index] / l_min
        if abs(max(min_l_dis) - l_max) <= l_max / 10:  # 如果某条最长边已经快延伸到相邻顶点时，则将距离置为1
            max_index = min_l_dis.index(max(min_l_dis))
            min_l_dis[max_index] = min_l_dis[max_index] / l_max
        if min_index >= 0 and max_index >= 0:
            continue
        if flag_debug:
            #print('[DEBUG fill_4points] box_point=', box[box_point_i][0], ', min_l=', min_l_dis, ', min_i=', min_i, ', min_i_points=', points[min_i[0]][0], points[min_i[1]][0])
            pass
        if min(min_l_dis) <= (l_min / 1.1) and max(min_l_dis) <= (l_max / 1.1):
            if (min_l_dis[0] * min_l_dis[1] / 2) <= s_rect / 10:
                direction_flag = True
                if flag_turn:
                    if image is None:
                        print('WARNING: fill_4points_turn image is none, cant display this image!')
                    else:
                        img = image.copy()
                        cv2.drawContours(img, box, -1, color_box, 2)  # 绘制最小面积外接矩形
                        cv2.drawContours(img, [box], -1, color_box, 0)
                        cv2.drawContours(img, [points], -1, color_line, 0)  # 绘制轮廓线
                        cv2.drawContours(img, points, -1, color_point, 3)
                        cv2.circle(img, (points[min_i[0]][0][0], points[min_i[0]][0][1]), 1, color_old, 3)  # 绘制规整示意点
                        cv2.circle(img, (points[min_i[1]][0][0], points[min_i[1]][0][1]), 1, color_old, 3)  # 绘制规整示意点
                        cv2.circle(img, (box[box_point_i][0][0], box[box_point_i][0][1]), 1, color_new, 3)
                        cv2.namedWindow('fill_4points_turn', 1)
                        cv2.resizeWindow("fill_4points_turn", 1024, 1024)
                        cv2.imshow("fill_4points_turn", img)
                        cv2.waitKey()
                if flag_debug:
                    print('[DEBUG fill_4points]\t\t\t{', min_i[0], points[min_i[0]][0], '+', min_i[1], points[min_i[1]][0], '} ->', box[box_point_i][0], end=' ')
                if min_index >= 0:
                    min_l_dis[min_index] = min_l_dis[min_index] * l_min
                if max_index >= 0:
                    min_l_dis[max_index] = min_l_dis[max_index] * l_max
                if min_l_dis[0] <= min_l_dis[1]:
                    points[min_i[0]][0][0] = box[box_point_i][0][0]
                    points[min_i[0]][0][1] = box[box_point_i][0][1]
                    # print(min_i[0], points[min_i[0]][0][0] != box[(len(box) + box_point_i - 1) % len(box)][0][0], points[min_i[0]][0][1] != box[(len(box) + box_point_i - 1) % len(box)][0][1])
                    # points_on_lines[box_point_i].append(min_i[1])
                    # points_on_lines[(len(box) + box_point_i - 1) % len(box)].append(min_i[1])
                else:
                    points[min_i[1]][0][0] = box[box_point_i][0][0]
                    points[min_i[1]][0][1] = box[box_point_i][0][1]
                    # print(min_i[1])
                    # points_on_lines[box_point_i].append(min_i[0])
                    # points_on_lines[(len(box) + box_point_i - 1) % len(box)].append(min_i[0])

                points_del_temp1 = []
                points_del_temp2 = []
                for pi in range(min(min_i) + 1, max(min_i)):
                    points_del_temp1.append(pi)
                for pi in range(max(min_i) + 1, len(points)):
                    points_del_temp2.append(pi)
                for pi in range(0, min(min_i)):
                    points_del_temp2.append(pi)
                if len(points_del_temp1) <= len(points_del_temp2):
                    points_del = points_del + points_del_temp1
                else:
                    points_del = points_del + points_del_temp2
                # if (min_i[1] - min_i[0]) * direction >= 0:  # 点的遍历方向相同
                #     for pi in range(min(min_i) + 1, max(min_i)):
                #         points_del.append(pi)
                # elif max(min_i) < len(points) - 1 or min(min_i) > 0:  # 遍历方向不同，但中间有跨越点
                #     for pi in range(max(min_i) + 1, len(points)):
                #         points_del.append(pi)
                #     for pi in range(0, min(min_i)):
                #         points_del.append(pi)
                # else:  # 方向相反时
                #     points_del.append(min_i[0])
                #     points_del.append(min_i[1])
                # points_del = list(set(points_del))
                # points_del.sort(reverse=True)  # 倒叙排序
                if flag_debug:
                    print(', then want to delete:', points_del)
                #print(len(points), (min_i[1] - min_i[0]), direction)
            else:  # 第一种失败：所缺面积过大
                if flag_debug:
                    print('[DEBUG fill_4points] fail 1! min_l_dis[0]*min_l_dis[1]/2=', min_l_dis[0] * min_l_dis[1] / 2, ', s_rect/10=', s_rect / 10)
                pass
        else:  # 第二种失败：最短距离太长了
            if flag_debug:
                print('[DEBUG fill_4points] fail 2! box_point=', box[box_point_i], ', min_l_dis=', min_l_dis, ', l_min=', l_min, ', l_max=', l_max)
            pass
    points_del = list(set(points_del))  # 删除待删除点中重复的点
    points_del.sort(reverse=True)  # 倒叙排序
    for p_i in points_del:  # 逐个删除
        if len(points) <= 4:  # 点太少了没必要删了
            if flag_debug:
                print('[DEBUG fill_4points] len(points) is too small, no need to delete points')
            break
        # switch = 0  # 异常控制按钮
        # for line_i in range(len(points_on_lines)):
        #     for line_p_i in points_on_lines[line_i]:
        #         if line_p_i == p_i:
        #             if len(points_on_lines[line_i]) <= 1:
        #                 switch = 1  # 边界上的点太少了，别删了
        #                 print('边界上的点太少了，别删了')
        #                 break
        #     if switch > 0:
        #         break
        # if switch == 1:
        #     continue
        flag = False
        for box_point_i in range(len(box)):  # 如果该点位于外接矩形顶点上，则不删除
            if points[p_i][0][0] == box[box_point_i][0][0] and points[p_i][0][1] == box[box_point_i][0][1]:
                flag = True
                break
        if flag:
            continue
        if flag_debug:
            print('[DEBUG fill_4points] delete points:', p_i, points[p_i])
        points = np.delete(points, p_i, axis=0)
    # print('fill_4points: %d -> %d' % (before_num, len(points)))
    if flag_image:
        if image is None:
            print('WARNING: fill_4points image is none, cant display this image!')
        else:
            img = image.copy()
            cv2.drawContours(img, box, -1, color_box, 2)  # 绘制最小面积外接矩形
            cv2.drawContours(img, [box], -1, color_box, 0)
            cv2.drawContours(img, [points], -1, color_line, 0)  # 绘制轮廓线
            cv2.drawContours(img, points, -1, color_point, 3)
            cv2.namedWindow('fill_4points', 1)
            cv2.resizeWindow("fill_4points", 1024, 1024)
            cv2.imshow("fill_4points", img)
            cv2.waitKey()
    return points


def cal_ang(point_1, point_2, point_3):
    """
    【计算角p1p2p3的角度值，范围是0°到180°】
    :param point_1: p1，1*2的ndarray
    :param point_2: p2，1*2的ndarray
    :param point_3: p3，1*2的ndarray
    :return: 角度值，float
    """
    a = math.sqrt((point_2[0][0]-point_3[0][0])*(point_2[0][0]-point_3[0][0])+(point_2[0][1]-point_3[0][1])*(point_2[0][1] - point_3[0][1]))
    b = math.sqrt((point_1[0][0]-point_3[0][0])*(point_1[0][0]-point_3[0][0])+(point_1[0][1]-point_3[0][1])*(point_1[0][1] - point_3[0][1]))
    c = math.sqrt((point_1[0][0]-point_2[0][0])*(point_1[0][0]-point_2[0][0])+(point_1[0][1]-point_2[0][1])*(point_1[0][1]-point_2[0][1]))
    A = math.degrees(math.acos((a*a-b*b-c*c)/(-2*b*c)))
    B = math.degrees(math.acos((b*b-a*a-c*c)/(-2*a*c)))
    C = math.degrees(math.acos((c*c-a*a-b*b)/(-2*a*b)))
    return B

def turn_to_right_angle(points, angle_1=10.0, angle_2=45.0, image=None, box=None, flag_image=True, flag_debug=False, flag_turn=False,
                        color_box=(255, 0, 0), color_point=(0, 0, 255), color_line=(0, 255, 0), color_old=(0, 127, 255), color_new=(0, 255, 255)):
    """
    【将锐角、钝角尽量修改为直角形】
    :param points: 轮廓点集，n*1*2的ndarray
    :param angle_1: 摇摆角度参数1，float
    :param angle_2: 摇摆角度参数2，float
    :param image: 展示图像（不修改原图像）
    :param box: 最小边界外接矩形的四个端点，4*1*2的ndarray
    :param flag_image: 处理后是否在image上展示图像
    :param flag_debug: 是否输出调试信息
    :param flag_turn: 是否逐个输出规整示意图
    :param color_box: 画图时的最小面积外接矩形的颜色
    :param color_point: 画图时的点的颜色
    :param color_line: 画图时的线的颜色
    :param color_old: 规整示意图中旧的点的颜色
    :param color_new: 规整示意图中规整后的点的颜色
    :return: 补全顶角后的点集，n*1*2的ndarray
    :return:
    """
    turn_count = 0  # 轮换计数器
    #while (angle < (90.0 - angle_1) or angle > (90.0 + angle_1)) and ((k[0] * 0.9 <= k_first <= k[0] * 1.1) or (k[1] * 0.9 <= k_first <= k[1] * 1.1)):  # 找到原轮廓上的第一个直角，且第一个斜率不能太离谱
    while True:
        flag = False
        for box_i in box:
            if points[0][0][0] == box_i[0][0] and points[0][0][1] == box_i[0][1]:
                flag = True
        if flag:
            break
        turn_temp = np.copy(points[0])
        points = np.delete(points, 0, axis=0)
        points = np.insert(points, len(points), turn_temp, axis=0)
        turn_count = turn_count + 1
        if turn_count > len(points) + 2:  # 轮换完还没有找到第一个直角
            # print('WARNING: turn_to_right_angle cant find the first right angle!')
            # break
            return points
    #print('fine')
    if box is None:
        box = get_min_area_rect(points)
    k = [0.1, 0.1, 0.1, 0.1]  # 4条边界线的斜率k，k=(y1-y2)/(x1-x2)
    b = [0.1, 0.1, 0.1, 0.1]  # 4条边界线的偏置b
    k[0] = (box[0][0][1] - box[1][0][1] + 0.000001) / (box[0][0][0] - box[1][0][0] + 0.000001)
    k[1] = (box[1][0][1] - box[2][0][1] + 0.000001) / (box[1][0][0] - box[2][0][0] + 0.000001)
    k[2] = k[0]
    k[3] = k[1]
    for l_i in range(len(box)):  # 计算4个偏置，b=y-kx
        b[l_i] = box[l_i][0][1] - k[l_i] * box[l_i][0][0]
    if flag_debug:
        print('[DEBUG turn_to_right_angle] global: len(points)=', len(points), ', box=', box[0][0], box[1][0], box[2][0], box[3][0], ', k=', k, ', b=', b)
    points_on_lines = []  # 轮廓上的点位于外接矩形的边上的点集
    for p_i in range(len(points)):  # 枚举轮廓上的每个点
        for l_i in range(len(k)):  # 判断该点是否位于哪条外接矩形的边直线上
            if abs((k[l_i] * points[p_i][0][0] + b[l_i]) - points[p_i][0][1]) <= 3 or abs((points[p_i][0][1] - b[l_i] + 0.000001) / (k[l_i] + 0.00001) - points[p_i][0][0]) <= 3:
                if flag_debug:
                    print('[DEBUG turn_to_right_angle] points_on_line: point %d (%d, %d) is on line %d : y=%f x + %f' % (p_i, points[p_i][0][0], points[p_i][0][1], l_i, k[l_i], b[l_i]))
                points_on_lines.append(p_i)  # 该点存在于边界线上，记录其下标
                break
    k = [k[0], k[1]]  # 4条边界线的斜率k，k=(y1-y2)/(x1-x2)，因为是矩形，所以斜率只有2个
    angle = cal_ang(points[-1], points[0], points[1])  # 第一个角度
    k_first = (points[0][0][1] - points[-1][0][1] + 0.000001) / (points[0][0][0] - points[-1][0][0] + 0.000001)  # 第一个斜率
    for p_i in range(len(points)):
        p1 = points[p_i]
        p2 = points[(p_i + 1) % len(points)]
        p3 = points[(p_i + 2) % len(points)]
        if flag_debug:
            print('[DEBUG turn_to_right_angle] on_line p2:', ((p_i + 1) % len(points)), p2[0], (((p_i + 1) % len(points)) in points_on_lines), end=' | ')
            print('p3:', ((p_i + 2) % len(points)), p3[0], (((p_i + 2) % len(points)) in points_on_lines))
        if ((p_i + 1) % len(points)) in points_on_lines and ((p_i + 2) % len(points)) not in points_on_lines:  # 如果p2在边界上，而p3不在，则将p3修正
            # print('如果p2在边界上，而p3不在，则将p3修正')
            angle = cal_ang(p1, p2, p3)  # 计算角p1p2p3的角度
            if ((90.0 - angle_2) < angle < 90.0) or (90.0 < angle < (90.0 + angle_2)):  # 角度合适，可以变为直角
                k1 = (p1[0][1] - p2[0][1] + 0.000001) / (p1[0][0] - p2[0][0] + 0.000001)  # 计算p1p2的斜率
                k_switch = 99999

                if k[0] * 0.9 <= k1 <= k[0] * 1.1:  # 判断p1p2的斜率属于哪条边
                    k_switch = 0
                    k1 = k[k_switch]
                elif k[1] * 0.9 <= k1 <= k[1] * 1.1:
                    k_switch = 1
                    k1 = k[k_switch]
                else:
                    # continue
                    pass
                b1 = p2[0][1] - p2[0][0] * k1
                k2 = -1 / (k1 + 0.000001)
                # k2 = k[(k_switch + 1) % len(k)]
                b2 = p2[0][1] - p2[0][0] * k2
                k3 = k1
                b3 = p3[0][1] - p3[0][0] * k3
                #print('k1 = %f, k2 = %f    k=%f %f' % (k1, k2, k[0], k[1]))
                if flag_debug:
                    print('[DEBUG turn_to_right_angle] angle=', angle, ', k=', k1, k2, 'points=', p1, p2, p3)
                # 直线1-p1p2   直线2-p1p2的垂线   直线3-经过p3的平行线
                if abs(k1 - 0.000001) <= 0.00001:
                    # print(k1, p1)
                    xp = p2[0][0]
                else:
                    xp = (b3 - b2 + 0.000001) / (k2 - k3 + 0.000001)
                yp = k2 * (b3 - b2) / (k2 - k3) + b2
                xmid = int((p3[0][0] + xp) / 2)
                ymid = int((p3[0][1] + yp) / 2)
                if flag_turn:
                    if image is None:
                        print('WARNING: turn_to_right_angle_turn image is none, cant display this image!')
                    else:
                        img = image.copy()
                        cv2.drawContours(img, box, -1, color_box, 2)  # 绘制最小面积外接矩形
                        cv2.drawContours(img, [box], -1, color_box, 0)
                        cv2.drawContours(img, [points], -1, color_line, 0)  # 绘制轮廓线
                        cv2.drawContours(img, points, -1, color_point, 3)
                        cv2.circle(img, (points[(p_i + 0) % len(points)][0][0], points[(p_i + 0) % len(points)][0][1]),
                                   1, color_old, 3)  # 绘制规整示意点
                        cv2.circle(img, (points[(p_i + 1) % len(points)][0][0], points[(p_i + 1) % len(points)][0][1]),
                                   1, color_old, 3)  # 绘制规整示意点
                        cv2.circle(img, (points[(p_i + 2) % len(points)][0][0], points[(p_i + 2) % len(points)][0][1]),
                                   1, color_old, 3)  # 绘制规整示意点
                        cv2.circle(img, (xmid, ymid), 1, color_new, 3)
                        cv2.namedWindow('turn_to_right_angle_turn', 1)
                        cv2.resizeWindow("turn_to_right_angle_turn", 1024, 1024)
                        cv2.imshow("turn_to_right_angle_turn", img)
                        cv2.waitKey()
                points[(p_i + 1) % len(points)][0][0] = points[(p_i + 1) % len(points)][0][0] - (xmid - points[(p_i + 2) % len(points)][0][0])
                points[(p_i + 1) % len(points)][0][1] = points[(p_i + 1) % len(points)][0][1] - (ymid - points[(p_i + 2) % len(points)][0][1])
                points[(p_i + 2) % len(points)][0][0] = xmid
                points[(p_i + 2) % len(points)][0][1] = ymid
            pass
        elif ((p_i + 1) % len(points)) not in points_on_lines and ((p_i + 2) % len(points)) in points_on_lines:  # 如果p2不在边界线上，而p3在边界线上，则将p2修正
            # print('如果p2不在边界线上，而p3在边界线上，则将p2修正')
            angle = cal_ang(p1, p2, p3)  # 计算角p1p2p3的角度
            if ((90.0 - angle_2) < angle < 90.0) or (90.0 < angle < (90.0 + angle_2)):  # 角度合适，可以变为直角
                k1 = (p1[0][1] - p2[0][1] + 0.000001) / (p1[0][0] - p2[0][0] + 0.000001)  # 计算p1p2的斜率
                k_switch = 99999
                if k[0] * 0.9 <= k1 <= k[0] * 1.1:  # 判断p1p2的斜率属于哪条边
                    k_switch = 0
                    k1 = k[k_switch]
                elif k[1] * 0.9 <= k1 <= k[1] * 1.1:
                    k_switch = 1
                    k1 = k[k_switch]
                else:
                    # continue
                    pass
                b1 = p2[0][1] - p2[0][0] * k1
                k2 = -1 / (k1 + 0.000001)
                # k2 = k[(k_switch + 1) % len(k)]
                b2 = p2[0][1] - p2[0][0] * k2
                k3 = k1
                b3 = p3[0][1] - p3[0][0] * k3
                # print('k1 = %f, k2 = %f    k=%f %f' % (k1, k2, k[0], k[1]))
                if flag_debug:
                    print('[DEBUG turn_to_right_angle] angle=', angle, ', k=', k1, k2, 'points=', p1, p2, p3)
                # 直线1-p1p2   直线2-p1p2的垂线   直线3-经过p3的平行线
                '''
                !!!!!!!!!!!!!!!!!!!!!!!斜率计算！
                '''
                if abs(k1 - 0.000001) <= 0.00001:
                    # print(k1, p1)
                    xp = p2[0][0]
                else:
                    xp = (b3 - b2 + 0.000001) / (k2 - k3 + 0.000001)
                yp = k2 * (b3 - b2) / (k2 - k3) + b2
                xmid = int((p3[0][0] + xp) / 2)
                ymid = int((p3[0][1] + yp) / 2)
                if flag_turn:
                    if image is None:
                        print('WARNING: turn_to_right_angle_turn image is none, cant display this image!')
                    else:
                        img = image.copy()
                        cv2.drawContours(img, box, -1, color_box, 2)  # 绘制最小面积外接矩形
                        cv2.drawContours(img, [box], -1, color_box, 0)
                        cv2.drawContours(img, [points], -1, color_line, 0)  # 绘制轮廓线
                        cv2.drawContours(img, points, -1, color_point, 3)
                        cv2.circle(img, (points[(p_i + 0) % len(points)][0][0], points[(p_i + 0) % len(points)][0][1]),
                                   1, color_old, 3)  # 绘制规整示意点
                        cv2.circle(img, (points[(p_i + 1) % len(points)][0][0], points[(p_i + 1) % len(points)][0][1]),
                                   1, color_old, 3)  # 绘制规整示意点
                        cv2.circle(img, (points[(p_i + 2) % len(points)][0][0], points[(p_i + 2) % len(points)][0][1]),
                                   1, color_old, 3)  # 绘制规整示意点
                        cv2.circle(img, (xmid, ymid), 1, color_new, 3)
                        cv2.namedWindow('turn_to_right_angle_turn', 1)
                        cv2.resizeWindow("turn_to_right_angle_turn", 1024, 1024)
                        cv2.imshow("turn_to_right_angle_turn", img)
                        cv2.waitKey()
                points[(p_i + 1) % len(points)][0][0] = points[(p_i + 1) % len(points)][0][0] - (
                            xmid - points[(p_i + 2) % len(points)][0][0])
                points[(p_i + 1) % len(points)][0][1] = points[(p_i + 1) % len(points)][0][1] - (
                            ymid - points[(p_i + 2) % len(points)][0][1])
                points[(p_i + 2) % len(points)][0][0] = xmid
                points[(p_i + 2) % len(points)][0][1] = ymid
            pass
        else:  # 其他情况不处理
            continue
    if flag_image:
        if image is None:
            print('WARNING: turn_to_right_angle image is none, cant display this image!')
        else:
            img = image.copy()
            cv2.drawContours(img, box, -1, color_box, 2)  # 绘制最小面积外接矩形
            cv2.drawContours(img, [box], -1, color_box, 0)
            cv2.drawContours(img, [points], -1, color_line, 0)  # 绘制轮廓线
            cv2.drawContours(img, points, -1, color_point, 3)
            cv2.namedWindow('turn_to_right_angle', 1)
            cv2.resizeWindow("turn_to_right_angle", 1024, 1024)
            cv2.imshow("turn_to_right_angle", img)
            cv2.waitKey()
    return points


def approx(polygon):
    points = polygon  # points就是轮廓点集

    box = get_min_area_rect(points)  # 得到该轮廓的最小面积外接矩形的4个端点
    s_build = cv2.contourArea(points)  # 得到mask的面积
    s_rect = cv2.contourArea(box)  # 得到最小边界外接矩形的面积
    if s_rect <= 2000 and s_build / s_rect >= 0.8:  # 如果mask本身就很小，而且所占的比例大于0.8，则直接取最小边界外接矩形为轮廓点
        points = box.copy()

    delta = 4  # 多边形初拟合阈值
    delta2 = 0.5  # Hausdorff距离规整阈值
    points = polygon_fitting(points, delta, image=None, flag_image=False)

    points = hausdorff_fitting(points, delta2, image=None, box=box, flag_image=False)  # Hausdorff距离深度规整
    points = polygon_fitting(points, delta, image=None, flag_image=False)  # 多边形初拟合

    points = fill_4points(points, image=None, box=box, flag_image=False)  # 补全四角
    points = polygon_fitting(points, delta, image=None, flag_image=False)  # 多边形初拟合

    if len(points) >= 6:  # 尝试将锐&钝角转化为直角
        points = turn_to_right_angle(points, image=None, box=box, flag_image=False)
    points = polygon_fitting(points, delta, image=None, flag_image=False)  # 多边形初拟合

    return points
