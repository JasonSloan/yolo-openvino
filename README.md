## 1. Environment Dependencies

### Switching Package Sources (Optional)

Use Tsinghua University mirrors from [here](https://mirrors.tuna.tsinghua.edu.cn/help/ubuntu/)

### Installing OpenVINO

Download OpenVINO Toolkit from [here](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html?VERSION=v_2023_3_0&OP_SYSTEM=LINUX&DISTRIBUTION=APT)

### Installing spdlog

```
apt-get install libspdlog-dev
```

### Adding Pre-Compiled OpenCV

Copy the pre-compiled opencv folder to your machine.

If you are using ubuntu, download the pre-built opencv4.2 library directly from [here](https://github.com/JasonSloan/DeepFusion/releases/download/v111/opencv4.2.tar), and put them into the  'yolo-openvino' directory, then set

```bash
export LD_LIBRARY_PATH=/path/to/opencv4.2/lib:$LD_LIBRARY_PATH
```

### Installing GPU Driver and Runtime (Version 24.09.28717.12)

Download and install the required driver and runtime from [here](https://github.com/intel/compute-runtime/releases) if you are going to use integreted graphics card or discrete graphics card manufactured by Intel as your inference backend.

### Setting GPU to Fixed Frequency (Optional)

1. Install Intel GPU tools:

   ```
   apt-get install intel-gpu-tools
   ```

2. Create a script to set GPU frequency:

   ```
   echo "intel_gpu_frequency -s 1600" >> /usr/local/bin/set_gpu_frequency.sh
   chmod 777 /usr/local/bin/set_gpu_frequency.sh
   ```

3. Create and enable a systemd service:

   ```
   sudo tee /etc/systemd/system/set_gpu_frequency.service > /dev/null << 'EOF'
   [Unit]
   Description=set_gpu_frequency
   After=network.target syslog.target
   Wants=network.target

   [Service]
   Type=simple
   ExecStart=bash /usr/local/bin/set_gpu_frequency.sh

   [Install]
   WantedBy=multi-user.target
   EOF

   systemctl daemon-reload
   systemctl enable set_gpu_frequency.service
   systemctl start set_gpu_frequency.service
   ```

------

## 2. Code Implementation

This repository contains object detection inference code for YOLOv5, YOLOv8, and YOLOv11, based on the Linux version of the OpenVINO inference framework.

### Features

- **Multi-threaded Asynchronous Inference**
  Inference is performed asynchronously using multiple threads.
- **Callback Functions for Results**
  Results are returned through callback functions.
- **Batch Inference Support**
  Supports both single-batch and multi-batch inference.

### Notes

- The current `CMakeLists.txt` file is configured to build a dynamic library.
- Test code for invoking inference can be found in the <test-model-infer> repository.

------

## Attribution

**Original Code**
If you use this code, please credit the source!