# 一. 环境依赖

换源(非必要):
https://mirrors.tuna.tsinghua.edu.cn/help/ubuntu/

安装openvino
https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html?VERSION=v_2023_3_0&OP_SYSTEM=LINUX&DISTRIBUTION=APT

安装spdlog:
apt-get install libspdlog-dev

将已编译好的opencv-build文件夹拷贝到机器中

安装24.09.28717.12版本显卡驱动以及runtime:
https://github.com/intel/compute-runtime/releases

设置GPU定频(非必要):
apt-get install intel-gpu-tools
echo "intel_gpu_frequency -s 1600" >> /usr/local/bin/set_gpu_frequency.sh
chmod 777 /usr/local/bin/set_gpu_frequency.sh

sudo tee /etc/systemd/system/set_gpu_frequency.service > /dev/null << 'EOF'
[Unit]
Description = set_gpu_frequency
After = network.target syslog.target
Wants = network.target
[Service]
Type = simple
ExecStart = bash /usr/local/bin/set_gpu_frequency.sh
[Install]
WantedBy = multi-user.target
EOF

systemctl daemon-reload
systemctl enable set_gpu_frequency.service
systemctl start set_gpu_frequency.service

# 二. 代码实现

基于linux版本openvino推理框架的yolov5、yolov8、yolov11的目标检测推理代码实现:

* 采用多线程异步推理的模式
* 通过回调函数返回结果
* 支持单batch推理、多batch推理



当前仓库中的CMakelists.txt为编译动态库的实现, 调用推理的测试代码见[test-model-infer]()



**!!原创代码, 引用请注明出处!!**