# StarPhotoPositioning
仅使用星星和拍摄时间来定位照片的拍摄地点  
Only use stars and time to locate the shooting location of photos  
## Features
- 一般只需**2颗星星**即可定位，图中需包含至少**2条铅垂线**
- **半自动**程序，需要通过拍摄时间的星图辨认星名，手动输入天文数据
- 输出拍摄地点的经维度格式的数据
- 拍摄时间精确在分钟为好，1小时的误差可能导致经维度10-20度的误差
- **误差一般在50km以下**
## Requirements
- python
- pyyaml
- opencv
- scipy
- sympy
- matplotlib
- sklearn
## Get Started
为查看对默认图像的定位效果，可直接运行
```bash
python calculate.py
```
## Tutorial
1. 将照片放入`images/`目录下，或者使用该目录下的默认照片
2. 在`config.yaml`中填写照片路径`image_path`，运行`stars.py`，用小框框选星星，并将得到的坐标填在`stars_img_pst`下
3. 通过[stellarium](https://stellarium.org)等天文软件查看拍摄时星空，辨认星星。在软件中设定观测点坐标为(0°N, 0°E)，读取星星的（时角，赤纬），按`stars_img_pst`的顺序填入`stars_sky_pst`
4. 查看`output/axis.png`，估计图中铅垂线的灭点（[Vanishing Point](https://en.wikipedia.org/wiki/Vanishing_point)）在图像坐标系的坐标，在`config.yaml`中修改`expected_center`和`expected_radius`。为排除某些伪铅垂线的干扰，设置`ignore_angle_scope`区间（规定指向照片上方为0°，顺时针旋转至下方为180°）以滤除此角度范围内的直线。运行`sky_top.py`以得到灭点的图中坐标，查看`output/sky_top.jpg`确认
5. 将灭点的图中坐标填入`sky_top_img_pst`，运行`calculate.py`得到拍摄地点地理位置
## Tips
- 一般使用`config.yaml`的默认参数即可获得良好效果，有特殊需求的情况下可参照opencv相关函数参数的含义来调整
- `is_fix_refraction_error`可用来修正大气折射误差，但运算量较大且修正幅度不高，不建议使用
- 该程序的主要误差一是灭点定位，二是照片畸变。如果你需要更好的效果，需通过专业软件修正照片畸变后再使用本程序
## How it works
- [仅凭星星和时间就能定位照片位置？](https://caveallegory.cn/2024/04/%E4%BB%85%E5%87%AD%E6%98%9F%E6%98%9F%E5%92%8C%E6%97%B6%E9%97%B4%E5%B0%B1%E8%83%BD%E5%AE%9A%E4%BD%8D%E7%85%A7%E7%89%87%E4%BD%8D%E7%BD%AE%EF%BC%9F/)
## Credits
- [天文学真的对个人来说，毫无用处吗？ - 鬼蝉的回答 - 知乎](
https://www.zhihu.com/question/603566190/answer/3313965267)
