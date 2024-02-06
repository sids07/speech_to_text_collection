from train_whisper.src.data_preprocessing import DataPreprocessing
from train_whisper.config.config import MODEL_NAME, TRAINING_ARGS
from train_whisper.src.evaluation_metrics import EvaluationMetrics
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

class ASRTrainer:
    
    def __init__(self):
        self.data_preprocessing = DataPreprocessing(
            model_name= MODEL_NAME
        )
                
        self.processor = self.data_preprocessing.get_processor()
        self.tokenizer = self.data_preprocessing.get_tokenizer()
        self.data = self.data_preprocessing.preprocess_data()
        self.data_collator = self.data_preprocessing.get_datacollator()
        
        self.evaluation_metrics = EvaluationMetrics(
            tokenizer= self.tokenizer
        )

        
        self.model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
        self.model.config.forced_decoder_ids = None
        self.model.config.suppress_tokens = []
        
    def trainer(self):
        
        training_args = Seq2SeqTrainingArguments(
            output_dir=TRAINING_ARGS["output_dir"],  # change to a repo name of your choice
            per_device_train_batch_size=TRAINING_ARGS["per_device_train_batch_size"],
            per_device_eval_batch_size=TRAINING_ARGS["per_device_eval_batch_size"],
            gradient_accumulation_steps=TRAINING_ARGS["gradient_accumulation_steps"],  # increase by 2x for every 2x decrease in batch size
            learning_rate=TRAINING_ARGS["learning_rate"],
            lr_scheduler_type= TRAINING_ARGS["lr_scheduler_type"],
            warmup_ratio=TRAINING_ARGS["warmup_ratio"],
            num_train_epochs =TRAINING_ARGS["num_train_epochs"],
            gradient_checkpointing=TRAINING_ARGS["gradient_checkpointing"],
            bf16=TRAINING_ARGS["bf16"],
            fp16=TRAINING_ARGS["fp16"],
            tf32=TRAINING_ARGS["tf32"],
            do_eval= TRAINING_ARGS["do_eval"],
            evaluation_strategy=TRAINING_ARGS["evaluation_strategy"],
            save_strategy=TRAINING_ARGS["save_strategy"],
            save_total_limit=TRAINING_ARGS["save_total_limit"],
            overwrite_output_dir = TRAINING_ARGS["overwrite_output_dir"],
            predict_with_generate=TRAINING_ARGS["predict_with_generate"],
            generation_max_length=TRAINING_ARGS["max_generation_len"],
            logging_steps=TRAINING_ARGS["logging_steps"],
            log_level=TRAINING_ARGS["log_level"],
            push_to_hub =TRAINING_ARGS["push_to_hub"],
            hub_model_id =TRAINING_ARGS["hub_model_id"],
            seed=TRAINING_ARGS["seed"],
            beta=TRAINING_ARGS["beta"]
        #     deepspeed="/kaggle/working/ds_config_zero3.json"
        )


        trainer = Seq2SeqTrainer(
            args=training_args,
            model=self.model,
            train_dataset=self.data["train"],
            eval_dataset=self.data["test"],
            data_collator=self.data_collator,
            compute_metrics=self.evaluation_metrics.compute_metrics,
            tokenizer=self.processor.feature_extractor,
        )
        
        train_result = trainer.train()
        
        metrics = train_result.metrics
        metrics["train_samples"] = len(self.data["train"])
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        if TRAINING_ARGS["do_eval"]:
            metrics = trainer.evaluate()
        
            metrics["eval_samples"] = len(self.data["test"])
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)
            
        trainer.save_model(TRAINING_ARGS["output_dir"])
        
        kwargs = {
            "finetuned_from": MODEL_NAME
        }
        
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(TRAINING_ARGS["output_dir"])

        if TRAINING_ARGS["push_to_hub"] is True:
            print("Pushing to hub...")
            trainer.push_to_hub()