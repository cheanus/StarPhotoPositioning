import cv2
import yaml
import numpy as np
from scipy.optimize import fsolve, root
from sky_top import cluster_center
from sympy import symbols, Eq, solve

def dms_to_rad(d, m, s, is_time=False):
    deg = d + m / 60 + s / 3600 if not is_time else (d + m / 60 + s / 3600)*15
    return deg * np.pi / 180

def delete_outliers(data):
    mean = np.mean(data)
    std = np.std(data)
    data = data[np.abs(data - mean) < 3 * std]
    return data

def refraction_pst(stars_sky_pst, stars_img_pst, sky_top_img_pst, img_center):
    # 由修正后的星空位置计算焦距
    s0 = get_focal_length(stars_sky_pst, stars_img_pst, img_center)
    stars_3d_pst1 = np.zeros((stars_img_pst.shape[0], 3), dtype=np.float64)

    for i in range(stars_img_pst.shape[0]):
        star_img_pst = stars_img_pst[i]
        
        focal_3d_pst = np.array([img_center[0], img_center[1], -s0], dtype=np.float64)
        star_3d_pst0 = np.hstack((star_img_pst, 0))

        star_focal_vector0 = star_3d_pst0 - focal_3d_pst
        sky_top_focal_vector = sky_top_img_pst - focal_3d_pst
        sky_top_star_vector = sky_top_img_pst - star_3d_pst0

        # 计算灭点处夹角
        alpha = np.arccos(
            sky_top_star_vector @ sky_top_focal_vector
            /np.linalg.norm(sky_top_star_vector) /np.linalg.norm(sky_top_focal_vector)
        )
        # 计算每颗星的天顶角
        theta0 = np.arccos(
            star_focal_vector0 @ sky_top_focal_vector
            /np.linalg.norm(star_focal_vector0) /np.linalg.norm(sky_top_focal_vector)
        )
        # 计算修正后的天顶角
        theta1 = theta0 - 1.02/np.tan((90-theta0+10.3/(90-theta0+5.11))/180*np.pi)/60/180*np.pi
        # 计算星处夹角
        beta = np.pi - alpha - theta0
        # 计算星与灭点的实际距离
        line_s = np.linalg.norm(sky_top_focal_vector)/np.sin(beta)*np.sin(theta1)
        # 计算实际星位
        unit_vector = star_3d_pst0 - sky_top_img_pst
        unit_vector /= np.linalg.norm(unit_vector)
        stars_3d_pst1[i] = sky_top_img_pst + unit_vector*line_s
    
    return stars_3d_pst1[:, :2]

def stars_sky_pst_formater(stars_sky_pst):
    for star_sky_pst in stars_sky_pst:
        star_sky_pst[0] = - dms_to_rad(*star_sky_pst[0], is_time=True)
        star_sky_pst[1] = dms_to_rad(*star_sky_pst[1])
    stars_sky_pst = np.array(stars_sky_pst)
    return stars_sky_pst

def get_focal_length(stars_sky_pst, stars_img_pst, img_center):
    focal_length_lists = []
    stars_img_rel_pst = stars_img_pst - img_center.reshape(1,-1)
    for i in range(stars_sky_pst.shape[0]):
        for j in range(i+1, stars_sky_pst.shape[0]):
            stars_cos = (
                np.cos(stars_sky_pst[i,1])*np.cos(stars_sky_pst[j,1])*np.cos(stars_sky_pst[i,0]-stars_sky_pst[j,0])
                + np.sin(stars_sky_pst[i,1])*np.sin(stars_sky_pst[j,1])
            )
            s = symbols('s', real=True, positive=True)
            eq1 = Eq(
                stars_img_rel_pst[i]@stars_img_rel_pst[j]+s**2,
                stars_cos*( ((stars_img_rel_pst[i]**2).sum()+s**2) * ((stars_img_rel_pst[j]**2).sum()+s**2) )**0.5
            )
            # 筛选正实数解
            s = solve(eq1, s)
            if len(s) >= 2:
                print('Warning: 求解焦距时找到两个及以上的正实数解。')
            focal_length_lists.extend([np.float64(sol) for sol in s])
    focal_length_lists = np.array(focal_length_lists)
    if len(focal_length_lists) > 2:
        focal_length_lists = delete_outliers(focal_length_lists).mean()
    elif len(focal_length_lists) == 2:
        focal_length_lists = focal_length_lists.max()
    else:
        focal_length_lists = focal_length_lists[0]
    return focal_length_lists

def geo2des(longitude, latitude):
    x = np.cos(latitude)*np.cos(longitude)
    y = np.cos(latitude)*np.sin(longitude)
    z = np.sin(latitude)
    return x, y, z

def des2geo(x, y, z):
    r = np.sqrt(x**2+y**2+z**2)
    longitude = np.arctan2(y, x)
    latitude = np.arcsin(z/r)
    return longitude, latitude

def get_location(stars_sky_pst, theta_cos):
    plain_args = []
    for i in range(stars_sky_pst.shape[0]):
        normal_vector = np.array(geo2des(*stars_sky_pst[i]))
        point_geo = stars_sky_pst[i].copy()
        point_geo[1] += np.arccos(theta_cos[i])
        point_des = np.array(geo2des(*point_geo))
        plain_args.append(np.hstack((normal_vector, -normal_vector@point_des)))
    plain_args = np.array(plain_args)

    location_lists = []
    for i in range(stars_sky_pst.shape[0]):
        for j in range(i+1, stars_sky_pst.shape[0]):
            tangent_vector = np.cross(plain_args[i, :3], plain_args[j, :3])
            point_online = np.linalg.inv(plain_args[[i,j], :2]) @ -plain_args[[i,j], 3]
            point_online = np.hstack((point_online, 0))

            t = symbols('t', real=True)
            eq = Eq(((point_online + t*tangent_vector)**2).sum(), 1)
            t = solve(eq, t)
            t = np.array(t, dtype=np.float64)

            for sol in t:
                point_des = point_online + sol*tangent_vector
                point_geo = des2geo(*point_des)
                location_lists.append(point_geo)
    location_lists = np.degrees(np.array(location_lists))
    if len(location_lists) > 2:
        location_lists = cluster_center(location_lists, 5)
    return location_lists

def main(sky_top_img_pst, stars_img_pst, img_center, stars_sky_pst, is_fix_refraction_error):
    stars_sky_pst = stars_sky_pst_formater(stars_sky_pst)

    def func(fix_stars_img_t):
        # 计算fix_stars_img_pst
        fix_stars_img_pst = stars_img_pst + (sky_top_img_pst[:2]-stars_img_pst)*fix_stars_img_t[:, np.newaxis]
        # 计算误差
        error = ((refraction_pst(stars_sky_pst, fix_stars_img_pst, sky_top_img_pst, img_center) - stars_img_pst)**2).sum(axis=1)
        print("迭代-大气折射误差："+str(error.sum()))
        return error
    
    if is_fix_refraction_error:
        stars_img_t = fsolve(func, np.zeros(stars_img_pst.shape[0]), xtol=1e-3, maxfev=20)
        stars_img_pst = stars_img_pst + (sky_top_img_pst[:2]-stars_img_pst)*stars_img_t[:, np.newaxis]

    s = get_focal_length(stars_sky_pst, stars_img_pst, img_center)
    focal_position = np.array([img_center[0], img_center[1], -s], dtype=np.float64)
    star_3d_positions = np.hstack((stars_img_pst, np.zeros((stars_img_pst.shape[0],1))))
    star_3d_vectors = star_3d_positions - focal_position.reshape(1,3)
    sky_top_3d_vector = sky_top_img_pst - focal_position

    theta_cos = star_3d_vectors@sky_top_3d_vector/(np.linalg.norm(star_3d_vectors, axis=1)*np.linalg.norm(sky_top_3d_vector))
    location_lists = get_location(stars_sky_pst, theta_cos)
    print('经度, 纬度')
    print(location_lists)

if __name__ =='__main__':
    args = yaml.safe_load(open("config.yaml"))
    args['sky_top_img_pst'].append(0)
    image = cv2.imread(args['img_path'])

    sky_top_img_pst = np.array(args['sky_top_img_pst'])
    stars_img_pst = np.array(args['stars_img_pst'])
    img_center = np.array(image.shape[-2::-1])/2
    stars_sky_pst = args['stars_sky_pst']
    is_fix_refraction_error = args['is_fix_refraction_error']
    
    main(sky_top_img_pst, stars_img_pst, img_center, stars_sky_pst, is_fix_refraction_error)