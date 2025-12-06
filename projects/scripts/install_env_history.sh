# ================================== 2025.11.25 ==================================
# 行不通，跑GRPO时候会挂掉，可能是驱动不支持（但忘了当时报错是什么了，也可能是flash-attn使用wheel安装然后GLIBC过高）
# 12.05验证了一下，跑单卡sft就挂掉了，报错应该是pytorch编译时的CUDA版本和驱动不兼容
conda create -n new python=3.11 -y
# 安装 CUDA Toolkit 和 NVCC 用于后续编译等
conda install -c nvidia/label/cuda-12.8.0 cuda-toolkit=12.8.0 -y
# 这一步只对当前终端有效，建议写入 ~/.bashrc 但为了稳妥先手动执行
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# 安装后续包
pip install vllm==0.11.0
pip install "sglang==0.5.4.post2"
pip install outlines-core==0.2.11
pip install "lmdeploy==0.10.2"
pip install "trl==0.23.1"
pip install autoawq -U --no-deps
pip install auto_gptq optimum bitsandbytes "gradio<5.33" -U
pip install transformers==4.57.1
pip install -e ".[all]"
pip install timm deepspeed==0.17.6
pip install qwen_vl_utils qwen_omni_utils keye_vl_utils -U
pip install decord librosa icecream soundfile -U
pip install liger_kernel nvitop pre-commit math_verify py-spy wandb swanlab -U
pip install flash-attn==2.8.1 --no-cache-dir --no-build-isolation
bash zsxm_train/test_sft.sh
bash zsxm_train/test_sft_ds.sh


# ================================== 2025.11.25 ==================================
# 尝试源码编译vllm，但是失败
conda create -n new python=3.11 -y
# 安装nvcc
conda install -c nvidia/label/cuda-12.4.1 cuda-toolkit -y
# 这一步只对当前终端有效，建议写入 ~/.bashrc 但为了稳妥先手动执行，好像不执行也行，nvcc -V的版本也是12.4，保险起见还是执行了两条命令
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH # 执行这个会影响git clone

# 安装pytorch
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.cuda.is_available())" # 必须输出 True，且不报 Error 200，才能进行下一步！
# 基于现有pytorch版本，源码构建vllm==0.11.0
git clone -b v0.11.0 --single-branch https://github.com/vllm-project/vllm.git
cd vllm
python use_existing_torch.py
pip install -r requirements/build.txt
# 在CMakeLists.txt的project(...)后面加上set(ENABLE_FP8 OFF CACHE BOOL "" FORCE)
pip install --no-build-isolation -e .


# ================================== 2025.12.04 ==================================
# 大部分功能都能用，就是Qwen3-VL无法训练GRPO
conda deactivate
conda remove -n new --all -y
conda create -n new python=3.10 -y
conda activate new
conda install -c nvidia/label/cuda-12.6.1 "cuda-toolkit=12.6.1"
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
pip install vllm==0.10
pip install "lmdeploy==0.11.0"
pip install "trl==0.23.1"
pip install autoawq --no-deps
pip install auto_gptq optimum bitsandbytes "gradio<5.33" -U
pip install transformers==4.57.1
pip install zstandard
pip install "sglang==0.5.6" --no-deps
pip install -e ".[all]"
#这是sglang的依赖但是没装pip install anthropic>=0.20.0 blobfile==3.0.0 build cuda-python decord2 flashinfer_cubin==0.5.3 flashinfer_python==0.5.3 grpcio-health-checking==1.75.1 grpcio-reflection==1.75.1 grpcio-tools==1.75.1 hf_transfer nvidia-cutlass-dsl==4.2.1 outlines==0.1.11 py-spy setproctitle sgl-kernel==0.3.18.post2 torch_memory_saver==0.0.9 torchao==0.9.0 torchcodec==0.7.0
pip install timm deepspeed==0.17.6
pip install qwen_vl_utils qwen_omni_utils keye_vl_utils -U
pip install decord librosa icecream soundfile -U
pip install liger_kernel nvitop pre-commit math_verify py-spy wandb swanlab -U
pip install flash-attn==2.8.3 --no-cache-dir --no-build-isolation

bash projects/scripts/env_test_sft.sh # Qwen2.5-VL-3B-Instruct sdpa Y Qwen3-VL-2B-Instruct sdpa Y
bash projects/scripts/env_test_sft_ds.sh # Qwen2.5-VL-7B-Instruct sdpa Y flash-attn Y Qwen3-VL-8B-Instruct sdpa Y flash-attn Y zero3 Y
bash projects/scripts/env_test_grpo_vllm.sh # Qwen/Qwen2.5-VL-3B-Instruct Y Qwen3-VL-2B-Instruct N


# ================================== 2025.12.05 ==================================
# 尝试直接安装，不编译vllm，从源码编译flash-attn。失败，deepspeed模式出现底层段错误
# 顺便验证2025.11.25的第一套方案是不是当时报错在flash-attn的安装上，但是验证失败
conda deactivate
conda remove -n new --all -y
conda create -n new python=3.10 -y
conda activate new
conda install -c nvidia/label/cuda-12.8.0 "cuda-toolkit=12.8.0"
pip install vllm==0.11.0 # 自动安装pytorch2.8.0+cu128
pip install "lmdeploy==0.11.0"
pip install "trl==0.23.1"
pip install autoawq --no-deps
pip install auto_gptq optimum bitsandbytes "gradio<5.33" -U
pip install transformers==4.57.1
pip install zstandard
pip install "sglang==0.5.6" --no-deps
pip install -e ".[all]"
pip install timm deepspeed==0.17.6
pip install qwen_vl_utils qwen_omni_utils keye_vl_utils decord librosa icecream soundfile liger_kernel nvitop pre-commit math_verify py-spy wandb swanlab -U
# 这里以后需要保证能连到github，建议开启proxyon
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout v2.8.3
python setup.py install

bash projects/scripts/env_test_sft.sh # Qwen2.5-VL-3B-Instruct sdpa Y Qwen3-VL-2B-Instruct sdpa Y
bash projects/scripts/env_test_sft_ds.sh # 报错，sdpa和更新deepspeed==0.18.2都无效
bash projects/scripts/env_test_grpo_vllm.sh # 同上



# ================================== 2025.12.06 ==================================
# 一次性跑通pytorch2.7.1+cu126，vllm0.12.0源码编译
conda create -n new python=3.10 -y
conda activate new
conda install -c nvidia/label/cuda-12.6.1 "cuda-toolkit=12.6.1"
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126

python -c "import torch; print(torch.__version__, torch.version.cuda)"
which nvcc
nvcc --version

# 检查没问题之后，下面正式开始安装vllm。先构建wheel，再安装。
git clone -b v0.12.0 --single-branch https://github.com/vllm-project/vllm.git
cd vllm
python use_existing_torch.py
pip install -r requirements/build.txt

# ***一步步修改编译所需要的环境变量***

# 1. 建基础目录
mkdir -p $CONDA_PREFIX/lib64

# 2. 把 targets 下真正的 include / lib 链接过来
cp -rsn $CONDA_PREFIX/targets/x86_64-linux/include/* $CONDA_PREFIX/include/
cp -rsn $CONDA_PREFIX/targets/x86_64-linux/lib/* $CONDA_PREFIX/lib/
cp -rsn $CONDA_PREFIX/targets/x86_64-linux/lib/* $CONDA_PREFIX/lib64/  # 为了兼容 FindCUDA 的 lib64 搜索
# 验证有输出
ls $CONDA_PREFIX/include | grep cuda.h
ls $CONDA_PREFIX/lib | grep libcudart
ls $CONDA_PREFIX/lib64 | grep libcudart

# 3. 设置相关环境变量
export CUDA_HOME=$CONDA_PREFIX
export CUDA_PATH=$CONDA_PREFIX
export CUDA_TOOLKIT_ROOT_DIR=$CONDA_PREFIX
export CUDAToolkit_ROOT=$CONDA_PREFIX

export CPATH=$CONDA_PREFIX/include:$CPATH
export LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$LD_LIBRARY_PATH
export CUDACXX=$CONDA_PREFIX/bin/nvcc

# 把有点迷惑的 NVCC_PREPEND_FLAGS 清掉，避免它乱填 ccbin
unset NVCC_PREPEND_FLAGS
unset NVCC_PREPEND_FLAGS_BACKUP

# 4. 安装git防止和后续编译下载东西时库冲突
conda install git
which git

# 5. 直接应用ChatGPT提供的方案B排除编译错误
cd $CONDA_PREFIX/lib/python3.10/site-packages/torch/share/cmake/Caffe2/public
cp cuda.cmake cuda.cmake.bak
# vim编辑cuda.cmake，在find_package(CUDAToolkit REQUIRED)这一行后面插入如下内容（包括注释）：
# "conda_envs/new"要根据环境该，一定要写绝对路径，不能用环境变量，否则ninja找不到（当然也有可能是我CMake代码问题）
vim cuda.cmake

# ZSXM Workaround: define CUDA::nvToolsExt if it was not created by FindCUDAToolkit
if(NOT TARGET CUDA::nvToolsExt)
  add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
  set_property(TARGET CUDA::nvToolsExt PROPERTY
    INTERFACE_LINK_LIBRARIES "/c22073/conda_envs/new/lib/libnvToolsExt.so.1"
  )
endif()

# 6. 返回vllm目录，开始编译
cd /c22073/codes/swift-new/vllm
proxyon # 开启代理防止git下载失败
# 在vllm文件夹下的CMakeLists.txt下搜索TORCH_SUPPORTED_VERSION_CUDA和TORCH_SUPPORTED_VERSION_ROCM并把版本改为2.7.1

pip install --no-build-isolation .

# 7. 后处理，手动找到install生成的wheel并存储起来
conda remove git
cd ..
rm -r vllm

# 然后重新创建一个终端，进行后续包的安装
pip install "lmdeploy==0.11.0"
pip install "trl==0.23.1"
pip install autoawq --no-deps
pip install auto_gptq optimum bitsandbytes "gradio<5.33" -U
pip install transformers==4.57.1
pip install zstandard pydantic==2.12.0
pip install "sglang==0.5.6" --no-deps
pip install -e ".[all]"
#这是sglang的依赖pip install anthropic>=0.20.0 blobfile==3.0.0 build cuda-python decord2 flashinfer_cubin==0.5.3 flashinfer_python==0.5.3 grpcio-health-checking==1.75.1 grpcio-reflection==1.75.1 grpcio-tools==1.75.1 hf_transfer nvidia-cutlass-dsl==4.2.1 outlines==0.1.11 py-spy setproctitle sgl-kernel==0.3.18.post2 torch_memory_saver==0.0.9 torchao==0.9.0 torchcodec==0.7.0
pip install timm deepspeed==0.17.6
pip install qwen_vl_utils qwen_omni_utils keye_vl_utils decord librosa icecream soundfile liger_kernel nvitop pre-commit math_verify py-spy wandb swanlab -U
# 源码安装flash-attn==2.8.3
git clone -b v2.8.3 --single-branch https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
python setup.py install
python -c "import flash_attn; print('flash_attn imported OK')"
pip show flash_attn
cd ..
rm -r flash-attention

bash projects/scripts/env_test_sft.sh # Qwen2.5-VL-3B-Instruct sdpa Y Qwen3-VL-2B-Instruct sdpa Y
bash projects/scripts/env_test_sft_ds.sh # Qwen2.5-VL-7B-Instruct flash-attn Y Qwen3-VL-8B-Instruct flash-attn Y zero3 Y
bash projects/scripts/env_test_grpo_vllm.sh # Qwen2.5-VL-7B-Instruct Y Qwen3-VL-2B-Instruct Y

# 在vllm0.11.2版本Qwen2.5-VLGRPO报错“This flash attention build does not support headdim not being a multiple of 32”
# 则运行unset VLLM_ATTENTION_BACKEND , 参考https://github.com/vllm-project/vllm/issues/26989 【试了不奏效】
# 另一个尝试export VLLM_USE_V1=0 ，参考https://github.com/modelscope/ms-swift/issues/6617 【试了也不奏效】
# 重装了vllm0.12.0，这个版本有修复这个bug，解决了问题。
# 或者设置VLLM_ATTENTION_BACKEND=FLASHINFER（Qwen2.5-VL、Qwen3-VL都不支持这个后端，不可行）
