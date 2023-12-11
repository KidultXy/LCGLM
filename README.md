# Local Knowledge Base Question-Answering System based on ChatGLM - LcGLM.

## Original Data Description

In terms of dataset construction, we referred to the ChineseNlpCorpus, a publicly available Chinese natural language processing corpus database. We selected 36,000 legal question-answer data sourced from Baidu Zhidao (Baidu Knows). Furthermore, we also consulted the open-source Lawyer LLaMA dataset and included data from the legal examination and legal instruction fine-tuning. Due to the size of the data, please click [here](https://pan.baidu.com/s/1WiDe5tLdR3IpKT63Ark9Qw) to obtain the data with the extraction code "3lph".

## Code File Description

- `clean_data.ipynb`: Python implementation for data cleaning and merging.
- `data`: Folder containing processed datasets and related information that can be used for model fine-tuning.
- `src`: Folder containing relevant code for model fine-tuning.
- `output`: Folder containing saved parameter information, log files, loss function graphs, etc. during model training.
- `eval_result`: Folder containing evaluation metrics and log files saved during model evaluation.
- `pre_result`: Folder containing prediction metrics, prediction results, and log files saved during model prediction.
- `requirements.txt`: File listing the third-party libraries and their versions on which this project depends.

## Running Instructions

To facilitate running the project, we have explicitly listed the overall command as follows:

### Environment Setup

```bash
git lfs install
git clone https://github.com/hiyouga/ChatGLM-Efficient-Tuning.git
conda create -n chatglm_etuning python=3.10
conda activate chatglm_etuning
cd ChatGLM-Efficient-Tuning
pip install -r requirements.txt
```

#### Download the ChatGLM2-6B model

To download the model from Hugging Face, you can use the following link: [chatglm2-6b](https://huggingface.co/THUDM/chatglm2-6b/tree/main) v1.0 (latest version as of August 19, 2023).

```bash
git clone https://huggingface.co/THUDM/chatglm2-6b
```

### Fine-tuning on a Single GPU (eg: Lora)

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

### Evaluation Metrics (BLEU & ROUGE Score)

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

### Model Prediction

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

### Browser Testing

```bash
python src/web_demo.py \
    --model_name_or_path /root/autodl-tmp/ChatGLM-Efficient-Tuning/chatglm2-6b \
    --finetuning_type lora \
    --checkpoint_dir /root/autodl-tmp/ChatGLM-Efficient-Tuning/output/lora/checkpoint-4000
```

## Other

If you have any questions, please contact [yixu.im@gmail.com](mailto:yixu.im@gmail.com).
