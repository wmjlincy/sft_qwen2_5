# 🚗 Qwen2.5-VL Autonomous Driving Fine-tuning

## ✨ Example Output

Below is an example of the model's output in a typical driving scenario:

<img width="1116" height="617" alt="image" src="https://github.com/user-attachments/assets/cbd51923-c0a3-4f60-ad16-544db1dc0410" />

> **Recommended speed:** 20 km/h  
> **Traffic light:** Currently in the straight lane, red light ahead. Need to stop and wait. (Note: Green light in the left-turn lane.)  
> **Traffic cones and obstacles:** None  
> **Crossroad:** There is a pedestrian crossing ahead, need to slow down.

---

## 🤖 Model Introduction

**Qwen2.5-VL** is the latest version of the Qwen series of large language models, specifically optimized and fine-tuned for vertical tasks in autonomous driving. This project enhances the base model using the efficient **LoRA (Low-Rank Adaptation)** technique to achieve precise adaptation for the following tasks:

- 🚦 Traffic light recognition and status alerts
- 🛑 Traffic cone and obstacle detection and warnings
- 📏 Adaptive speed recommendation

### 🔧 Technical Features

- **Efficient Fine-tuning**: Trainable low-rank matrices are injected only into key projection layers (e.g., `q_proj`, `v_proj`), significantly reducing the number of parameters.
- **Low Parameter Count**: Trainable parameters account for only **0.1% – 1%** of the original model, greatly reducing computational overhead and training time.
- **Strong Generalization**: Maintains the model's original general capabilities while adapting to new tasks.
- **Deployment-Friendly**: The fine-tuned model size remains almost unchanged, facilitating integration and deployment.

---

## ⚙️ Requirements

To ensure optimal compatibility and performance, please make sure your environment meets the following requirements:

- **Python** ≥ 3.9
- **PyTorch** ≥ 2.1
- **Transformers** library (recommended to use the latest version)

---

## 🚀 Quick Start

Run the following command to start the fine-tuning process:

```bash
python sft_qwen2_5.py \
  --model_dir Qwen2.5-VL-7B-Instruct \
  --output_dir model_fine_tune/Qwen2.5-VL-7B-Instruct-LoRA \
  --data_path data/labelresult_image_2dtlr.txt \
  --img_size 960 540 \
  --train_epochs 5 \
  --batch_size 16



