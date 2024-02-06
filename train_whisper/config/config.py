DATA_OVERVIEW="data/overview-of-recordings.csv"
MODEL_NAME="openai/whisper-base"

TRAINING_ARGS={
    "output_dir":"whisper_finetune",
    "per_device_train_batch_size":4,
    "per_device_eval_batch_size":4,
    "gradient_accumulation_steps":4,
    "learning_rate":1e-5,
    "warmup_ratio":0.05,
    "num_train_epochs":5,
    "gradient_checkpointing": True,
    "bf16": True,
    "fp16":False,
    "tf32":True,
    "evaluation_strategy":"epoch",
    "push_to_hub":True,
    "save_strategy":"no",
    "save_total_limit":None,
    "hub_model_id":"whisper_finetune",
    "do_eval":True,
    "logging_steps":15,
    "log_level":"info",
    "max_generation_len":225,
    "predict_with_generate":True,
    "overwrite_output_dir": True,
    "lr_scheduler_type":"cosine",
    "seed": 42,
    "beta":0.1
}