453cases 三模型重建适配器（第二版：三个模型各跑原图与 remap）
=================================

本版针对 case/conf/cam_N.yaml + case/picture/cam_N/数字.jpg。
本机没有你的业务数据。已知格式可直接运行；业务 YAML 格式未知时会停止并给出字段问题，
必须先确认配置结构，不能把测试通过等同于已适配你的标定。
本目录仅包含适配代码、示例配置、测试及说明，不包含模型代码、权重或业务数据。

1. 先查看配置结构，不必粘贴标定数值
---------------------------------
在服务器进入本目录，激活一个有 PyYAML 的环境。inspect_schema.py 是独立脚本，
不需要其他项目文件、NumPy、Pillow 或模型环境：

  python inspect_schema.py \
    --root /path/to/453cases \
    --out schema_report.json

会选择根目录下第一个含 conf/ 和 picture/ 的场景，检查其所有 cam_*.yaml。
默认只打印字段名、类型、列表长度；不打印标定值、图片名、场景名或完整路径。
把 schema_report.json 的内容发回即可。若需确认投影名称且可以分享型号字段：

  python inspect_schema.py --case /path/to/453cases/具体场景 \
    --include-model-names --out schema_models.json

这个可选参数只额外打印 model/model_type/camera_model/projection_model/distortion_model 的值。
只知道字段名仍可能无法判断厂商数字枚举含义；此时需要模型名称或枚举定义，不能靠猜。

2. 配置现有安装路径
-----------------
  cp paths.example.json paths.json
  nano paths.json

分别填写三个环境的 python 绝对路径、代码目录、权重路径。
三个环境不需要合并。若 OmniVGGT 和 Any3R 的 Conda 环境分别叫 omnivggt、any3r，查询路径：
  conda run -n omnivggt python -c "import sys; print(sys.executable)"
  conda run -n any3r python -c "import sys; print(sys.executable)"
在平时能够运行 Pi3X 的本机 Python 环境中查询：
  python -c "import sys; print(sys.executable)"
将输出分别填到对应模型的 python 字段。run_all.py 只负责调度，推理使用各自的 Python。
Pi3X 是 Pi3 仓库中 pi3/models/pi3x.py 的模型，不是旧版 Pi3。
Any3R checkpoint 填目录，其中必须同时有 config.json 和 model.safetensors。
Pi3X、OmniVGGT checkpoint 填完整 .safetensors 文件路径。

本适配器使用 Any3R 仓库的 any3r/cameras.py 进行统一投影和 remap，三个模型运行时都
需要配置 any3r.repo。只加载其 NumPy 几何模块；Pi3X 环境不用导入 Any3R 网络。

新增依赖：numpy、Pillow、PyYAML、OpenCV；评测和测试额外使用 scipy。
先在各环境检查，已有依赖就不必重装，也不需要重新安装 torch：

  /你的环境/bin/python -c 'import numpy, PIL, yaml, cv2'

缺什么才补什么，例如：
  /你的环境/bin/python -m pip install PyYAML
  /你的环境/bin/python -m pip install --no-deps opencv-python-headless==4.11.0.86
  /评测环境/bin/python -m pip install scipy

上面的 OpenCV 版本在本机测试过。已有可用 cv2 时不要再叠装第二种 OpenCV 包。
只用 inspect_schema.py 不需要 OpenCV、GPU 或权重。Python 需要 3.10 或更新版本。

OmniVGGT 的上游初始化会尝试额外下载 DINOv2；完整权重已经包含它。应用唯一的上游补丁：

  python patch_omnivggt_offline.py /你的代码目录/OmniVGGT

补丁修改 omnivggt/models/aggregator.py 的 _build_patch_embed 初始化分支，
增加 OMNIVGGT_SKIP_BACKBONE_DOWNLOAD=1 开关；原默认行为不变。
补丁只适用于与已测试代码片段兼容的版本，找不到片段会报错，不会模糊匹配乱改。
run.py 设置开关并 strict=True 加载完整 checkpoint；缺参数立即报错。
不修改 Pi3X、Any3R 上游网络。不会改你的 conda 源、代理设置或原始数据。

3. 先选帧、判断鱼眼，不加载模型
----------------------------
  python run.py \
    --case /path/to/453cases/具体场景 \
    --inspect-only --out /path/to/reconstruction/inspect_001

默认扫描该场景已有的所有 picture/cam_N/，取它们共同拥有的最小数字编号。
数字以整数精确比较，不转浮点数，长时间戳不会丢精度；前导零不影响比较。
同一相机下若两个文件映射到同一数字，会报歧义错误。任何所选相机没有共同帧都会停止。
不默默跳过缺帧相机，也不分别取各相机的第一张凑在一起。
纯数字文件名直接适用。混合名称默认提取末尾数字；其他规则可指定：
  --frame-regex '^(?P<id>\d+)$'
文件名编号相同不自动证明曝光同步，inspection.json 明确记录 synchronization_verified=false。
如果数字是时间戳、但各相机时间略有偏差，需要之后加入有明确定义的同步容差；本版只做精确编号匹配。

终端逐相机显示 fisheye=YES / NO / UNKNOWN 和判断依据。
模型名、畸变声明不足或相互矛盾时不默认当作 pinhole。
inspection.json 包含选图路径和归一化内参，属于本地运行记录，不要放进公开仓库。
若只想分享结构，请分享第 1 步的 schema_report.json。

4. 一条命令跑三个模型、六种配置
--------------------
初次建议选 2～3 个有重叠视野的相机，以 224 分辨率检查输入和显存。
这些相机应观察同一静态区域；互不重叠的环视相机不保证能从图像恢复可靠相对位姿。

  python run_all.py --config paths.json \
    --case /path/to/453cases/具体场景 \
    --cameras 0,1,2 --size 224 \
    --directions='0,0;-50,0;50,0' \
    --out-root /path/to/reconstruction/case001_test --gpu 0

依次调用每个模型自己的 Python 环境，不需要混用三个环境：
  pi3x_raw       原始鱼眼/普通图；原始鱼眼不传伪造的透视内参。
  pi3x_remap     每个鱼眼拆成多个透视视角，传虚拟透视内参。
  omnivggt_raw   原始图基线；满足条件的透视图才能传相机先验。
  omnivggt_remap 多方向透视图；有明确位姿且满足条件时传相机先验，否则使用透视 RGB。
  any3r_raw      原始图和标定射线图，支持真实鱼眼投影。
  any3r_remap    多方向透视图和对应透视射线图，用于对比重采样及视场拆分的影响。

三个模型均会运行，两种模式共六个独立结果目录，每个都有点云、预览和指标。
Any3R 原图模式本身支持标定鱼眼；remap 是额外对照，不是使用 Any3R 的必要步骤。

--cameras 去掉后使用所有现有 cam_N 文件夹；--size 448 提高分辨率，会增加显存。
默认每个鱼眼拆 5 个方向（中、左、右、上、下）；上例用 3 个方向降低测试显存。
13 路鱼眼默认可展开到 65 张，不能假设小显存 GPU 能一次跑完。优先减相机/方向/分辨率。
size 要是 112 的正整数倍（兼容 14/16 patch size）。输入保持宽高比缩放，再居中补黑到方形。
黑色实物表面不会因颜色暗而被过滤；只按几何有效区域及置信度过滤。

选一个具体的共同编号：
  在上述命令加入 --frame-id 1730300000123456789

只跑某个模型的两个模式，或只跑某种配置：
  --runs pi3x:raw,pi3x:remap
  --runs omnivggt:raw,omnivggt:remap
  --runs any3r:raw,any3r:remap
  --runs omnivggt:remap

如果想控制先验带来的差异，可加 --no-camera-priors 关闭 Pi3X/OmniVGGT 相机条件。
Any3R 仍使用射线图，否则会改变其输入相机假设。

5. 单张图片
-----------
  python run_all.py --config paths.json \
    --image /path/to/453cases/具体场景/picture/cam_3/1730300000123456789.jpg \
    --size 224 --out-root /path/to/reconstruction/single001 --gpu 0

程序从父文件夹识别 cam_3，自动找同场景 conf/cam_3.yaml。
--image 与 --case / --cameras / --frame-id 不混用。
单图输出是模型先验驱动的几何估计。把一张鱼眼 remap 成多图并没有增加真实视差，
也不保证模型预测的多个虚拟视角具有严格一致的共同光心；本版保存虚拟旋转供检查，
不会把预测点云称为已验证的严格刚体多目重建。

6. 单独运行 / 只看 remap
----------------------
  /Pi3X环境/bin/python run.py --config paths.json --model pi3x --mode remap \
    --case /path/to/453cases/具体场景 \
    --cameras 0,1 --size 224 --out /path/to/reconstruction/pi3x_remap001

先不加载模型，只导出输入图和掩码（可以不写 --model）：
  /有OpenCV的环境/bin/python run.py --config paths.json --prepare-only --mode remap \
    --case /path/to/453cases/具体场景 \
    --cameras 0 --size 448 --out /path/to/reconstruction/remap_check001

--fov 100 是每张方形虚拟透视图的水平/垂直视场角。
--directions='0,0;-50,0;50,0;0,-50;0,50' 是 yaw,pitch（度）；正 yaw 向右，正 pitch 向上。
畸变普通相机在 remap 模式只产生正前方一张去畸变透视图；已经无畸变的 pinhole 保留原视图。
检查投影可逆性、图像边界、镜头有效域；覆盖率低于 --min-coverage 0.25 的视图会打印并跳过。
不会通过填充虚构像素扩大鱼眼实际视场。

7. 每次运行的输出
-----------------
  inspection.json       文件选择、相机判断、标定信息、配置 SHA256
  inputs/*.png           真正送入模型的 RGB 图与 *_mask.png 有效区域
  views.json             每个视图来源、K、缩放补边、虚拟方向、T_source_virtual
  camera_inputs.npz       缩放后 K、射线图、有效掩码
  points.ply             带 RGB 的完整过滤后点云（CloudCompare / MeshLab 等可打开）
  points.npy             同一批点的 [N,3] 数组，供后续真实点云评测
  viewer.html            可离线打开的交互点云预览：拖动旋转、滚轮缩放、双击复位
  predictions.npz        稠密点、置信度、预测相机位姿、径向距离、掩码和 RGB
  metrics.json           有效比例、点数、距离分布、射线角度偏差等诊断
  run.json               模型路径、版本、显存/耗时、是否实际使用内参/位姿、脚本哈希

viewer.html 显示固定随机抽样最多 4 万点，仅显示坐标被归一化，points.ply 不做显示归一化。
将 viewer.html 下载到本地浏览器直接打开即可，无需服务器开放端口或联网下载 JS。
点云过滤默认每视图丢弃最低 20% 置信度；不同模型置信度数值不可直接比较。
Any3R 的射线角度接近零是结构保证，不代表它的真实重建精度最好。
点数和推理耗时也不是几何精度；单次未预热耗时不是公平速度基准。
场景尺寸、相机间距和深度绝对单位尚未验证，不要直接作为米制测量结果。
所有输出目录必须新建或为空，并且在原 case 之外，防止覆盖数据/旧结果。

8. 有真实点云后评测
-----------------
当前无真实点云，因此默认不计算伪造的 Chamfer/ATE/PSNR。
已提供点云评测入口（GT 和预测都先准备为 .npy [N,3]）：

  python evaluate_cloud.py --pred /输出目录/points.npy --gt /真实点云/gt.npy \
    --pred-to-gt /已确认的坐标变换/T.npy --units m --threshold 0.1 \
    --out /输出目录/gt_metrics.json

T.npy 是明确给定的 4x4 预测坐标 -> GT 坐标变换，可包含统一正尺度。
只有确认两者坐标系和尺度完全相同时，才用 --same-frame-and-scale 替代 --pred-to-gt。
输出两个方向最近点距离、双向平均距离、双向平均平方距离、阈值 precision/recall/F1。
默认每云固定种子采样最多 20 万点，记录采样数量和单位。没有自动 ICP 或偷偷优化尺度。
必须使用可比较的可见区域；遮挡、视野范围和点密度会影响这些指标。
深度 GT 的径向/轴向约定、位姿 GT 的尺度及坐标系、业务点云格式需要确认后再接入。

9. YAML 解析约定及需要修改的位置
------------------------------
dataset.py::normalize 是业务 YAML 的适配入口；发回结构信息后主要修改这里。
当前支持明确的 pinhole/perspective、OpenCV Brown-Conrady/radtan/plumb_bob、
OpenCV fisheye/KB4、Kalibr pinhole+equidistant、camodocal KANNALA_BRANDT。
K 支持 3x3/list/OpenCV-matrix/ROS camera_matrix.data；
内参向量必须明确为 [fx,fy,cx,cy]；分辨率必须与图片匹配，不自动猜裁剪/缩放。
camodocal KB 的 k2,k3,k4,k5 会转换为 OpenCV KB4 的四个系数。
仅写 fisheye、未知数字枚举、MEI/OCAM 等未经确认的组合会停止，不能据 D 长度推断模型。
Any3R 上游可支持更多投影，但本数据适配器不会在缺少参数语义时盲目套用。
默认像素中心遵循 OpenCV 整数坐标；规范配置可设 pixel_convention: colmap 明确转换。
可选 valid_fov_deg 限定镜头有效角域；缺省时只检查投影可逆性，并报告实际镜头边界未知。

无需修改原 YAML，也可用 --calibration-overrides /你的本地/calibration.json 覆盖某些相机。
格式参考 calibration.example.json；里面是示例数字，禁止当作真实标定直接运行业务数据。
覆盖映射的 key 是 cam_0 等；每项必须提供完整、已验证的规范标定。
其他相机继续读原 YAML。覆盖是否使用会记入 inspection.json。

只自动接受明确的 T_world_camera（相机坐标到共同世界/车体坐标的 4x4 刚体变换），
且必须同时声明 camera_axes: opencv、translation_unit: m。
这里 world 可以是同一时刻共同车体坐标，只要所有相机一致。
不会猜 T_cam_imu / extrinsics / rvec 的方向或毫米/米单位。
若原配置是 world-to-camera，必须在确认后取逆转换；如果轴定义不同，还需正确变换轴。

geometry.py：投影、鱼眼多方向 remap、K 缩放、补边、掩码；
run.py：三个模型的条件输入和输出适配；
outputs.py / evaluate_cloud.py：点云、离线预览、诊断及真实点云评测；
run_all.py：分别调用现有环境顺序运行，某个失败会保留日志并继续其他模型。

10. 模型接口限制与 3DGS
----------------------
Pi3X：RGB + 可选 K/rays/depth/camera-to-world。其 ray 条件内部会除以 z，
且输出本地深度为正 z，不能由“参数名 rays”推断它原生支持超过 180 度鱼眼。
原鱼眼/未去畸变图在本版仅作为 RGB 基线；remap 透视图才注入正确 K。
Pi3X 内部像素网格是 j+0.5，本版在传 K 前补偿半像素，保持与 remap 几何一致。
至少两个视图有明确且可用的位姿时才注入位姿条件。

OmniVGGT：相机条件同时需要外参和内参，编码是旋转/平移/两维 FoV。
没有仅 K 的独立接口，且编码不包含畸变或偏心主点；仅中心附近的透视 K+明确位姿会注入。
其余情况打印未使用条件及原因，保留 RGB 基线。可使用 omnivggt:remap 检查透视版本。
没有把相机名里的 Omni 当作原生鱼眼校准支持的证据。

Any3R：RGB + 标定单位射线图，包含鱼眼投影、内参和畸变。
已安装公开版没有已知位姿输入参数；位姿由模型预测。

检查的三个公开代码版本都没有原生 3D Gaussian Splatting 输出头：
  Pi3 9fa3ddb3f8d53041f8b2738df404f62223bbaa7b
  OmniVGGT c8cbfc6bcd57a07bf08a1f97804da9827847af8c
  Any3R c6c57bdda7b2526bdf6be28cc3b71d5c56a1eb15
因此本版不生成“3DGS 版本”，普通点云 PLY 不包含 Gaussian 的尺度/旋转/不透明度/SH 属性。
后续如需 3DGS，可另接训练/拟合流程，并处理标定、可见视域、曝光与相机位姿；不能只改扩展名。

11. 验证与公开仓库
-----------------
  ANY3R_REPO=/你的代码目录/Any3R python -m unittest -v test_adapter

用包含 scipy/OpenCV 的环境运行。模型 GPU 冒烟测试见 VALIDATION.txt，
公开样例转换出的测试目录不表示真实同步环视数据。尚未验证业务 YAML 或业务重建质量。

目标仓库：https://github.com/happyevery365/resulttest
在另一台服务器获取代码：
  git clone https://github.com/happyevery365/resulttest.git
  cd resulttest
公开时只放本目录代码和示例；不放 paths.json、业务标定、图片、输出、权重、环境或个人邮箱。
上游代码及权重由各自许可证约束，本适配包不重新分发它们。
