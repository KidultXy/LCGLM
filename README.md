# 基于 ChatGLM 的本地知识库知识问答系统——LawGLM

## 原始数据说明

数据集的构造上，我们参考ChineseNlpCorpus 公开中文自然语言处理语料数据库，选择来其中来源于百度知道的和法律相关的3.6 万条法律问答数据；此外，我们参考了开源的Lawyer LLaMA数据集，增加法考数据和法律指令微调数据。因为数据大小原因，点击[链接](https://pan.baidu.com/s/1WiDe5tLdR3IpKT63Ark9Qw )获取数据，提取码为 3lph 。

## 代码文件说明

`clean_data.ipynb`为对数据集进行清洗、合并的python实现；
`data`文件存放处理完毕、可用于模型微调的数据集及其相关信息；
`src`文件存放模型微调的相关代码；
`output`文件存放模型训练时保存的参数信息、日志文件、损失函数图像等；
`eval_result`存放模型评估时保存的评估指标与日志文件；
`pre_result`存放模型预测时保存的预测指标、预测结果与日志文件；
`requirements.txt`为该项目所依赖的第三方库及其版本号。

## 运行说明

为方便运行，我们将整体运行命令明确列出如下：

### 环境搭建

```bash
git lfs install
git clone https://github.com/hiyouga/ChatGLM-Efficient-Tuning.git
conda create -n chatglm_etuning python=3.10
conda activate chatglm_etuning
cd ChatGLM-Efficient-Tuning
pip install -r requirements.txt
```

#### 下载ChatGLM2-6B模型

从huggingface上下载链接，版本[chatglm2-6b](https://huggingface.co/THUDM/chatglm2-6b/tree/main) v1.0（截至2023/8/19最新版本）

```bash
git clone https://huggingface.co/THUDM/chatglm2-6b
```

### 单 GPU 微调训练（以lora为例）

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --model_name_or_path /root/autodl-tmp/ChatGLM-Efficient-Tuning/chatglm2-6b \
    --do_train \
    --dataset law_data \
    --finetuning_type lora \
    --output_dir /root/autodl-tmp/ChatGLM-Efficient-Tuning/output/lora \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss True \
    --fp16
```

### 指标评估（BLEU分数和汉语ROUGE分数）

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --model_name_or_path /root/autodl-tmp/ChatGLM-Efficient-Tuning/chatglm2-6b \
    --do_eval \
    --dataset law_data \
    --finetuning_type lora \
    --checkpoint_dir /root/autodl-tmp/ChatGLM-Efficient-Tuning/output/lora/checkpoint-4000 \
    --output_dir /root/autodl-tmp/ChatGLM-Efficient-Tuning/eval_result/lora \
    --per_device_eval_batch_size 8 \
    --max_samples 100 \
    --predict_with_generate
```

### 模型预测

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --model_name_or_path /root/autodl-tmp/ChatGLM-Efficient-Tuning/chatglm2-6b \
    --do_predict \
    --dataset law_data \
    --finetuning_type lora \
    --checkpoint_dir /root/autodl-tmp/ChatGLM-Efficient-Tuning/output/lora/checkpoint-4000 \
    --output_dir /root/autodl-tmp/ChatGLM-Efficient-Tuning/pre_result/lora \
    --per_device_eval_batch_size 8 \
    --max_samples 100 \
    --predict_with_generate
```

### 浏览器测试

```bash
python src/web_demo.py \
    --model_name_or_path /root/autodl-tmp/ChatGLM-Efficient-Tuning/chatglm2-6b \
    --finetuning_type lora \
    --checkpoint_dir /root/autodl-tmp/ChatGLM-Efficient-Tuning/output/lora/checkpoint-4000
```

## 其他

如有问题，请联系yixu.im@gmail.com

