Model Introduction: 

Qwen2.5-VL is the latest release in the Qwen series of large language models (LLMs). To enhance its performance on vertical tasks in autonomous driving, we have fine-tuned Qwen2.5 for specific applications, including traffic light recognition and alerts, detection and warning of obstacles such as traffic cones, and adaptive speed recommendation. The fine-tuning leverages the efficient LoRA (Low-Rank Adaptation) technique, which injects trainable low-rank matrices only into key projection layers (e.g., q_proj, v_proj) of the model. This approach adapts the model to new tasks with a minimal number of trainable parameters (typically only 0.1%–1% of the original model's parameters), significantly reducing computational overhead, accelerating training, preserving the model's general capabilities, and greatly simplifying deployment.



Requirements: 

We recommend using the latest version of the transformers library to ensure compatibility and optimal performance. This project requires Python 3.9 or higher and relies on the PyTorch 2.1+ framework.



Quick Start: 

You can quickly launch the fine-tuning process with the following command:

bash
python sft_qwen2_5.py \
  --model_dir Qwen2.5-VL-7B-Instruct \
  --output_dir model_fine_tune/Qwen2.5-VL-7B-Instruct-LoRA \
  --data_path data/labelresult_image_2dtlr.txt \
  --img_size 960 540 \
  --train_epochs 5 \
  --batch_size 16
