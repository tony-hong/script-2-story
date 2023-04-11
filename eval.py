import pandas as pd
import numpy as np
import torch
import pyarrow as pa
import pyarrow.dataset as ds
import pandas as pd
from datasets import Dataset
from datasets import load_dataset, load_metric
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

# load rouge
rouge = load_metric("rouge")

# load testset
test_set = pd.read_csv("../../data/valid_val_df.csv")
### convert to Huggingface dataset
test_set = test_set.dropna()
test_set = Dataset.from_pandas(test_set)

# load tokenizer

tokenizer = AutoTokenizer.from_pretrained("allenai/led-large-16384")
led = AutoModelForSeq2SeqLM.from_pretrained("allenai/led-large-16384", gradient_checkpointing=False, use_cache=False)

# load tokenizer
model = led.to("cuda").half()

def generate_answer(batch):
    inputs_dict = tokenizer(batch["source"], padding="max_length", max_length=8192, return_tensors="pt", truncation=True)
    input_ids = inputs_dict.input_ids.to("cuda")
    attention_mask = inputs_dict.attention_mask.to("cuda")
    global_attention_mask = torch.zeros_like(attention_mask)
    # put global attention on <s> token
    global_attention_mask[:, 0] = 1

    predicted_summary_ids = model.generate(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
    batch["predicted_summary"] = tokenizer.batch_decode(predicted_summary_ids, skip_special_tokens=True)
    return batch


result = test_set.map(generate_answer, batched=True, batch_size=8)

print("Result:", rouge.compute(predictions=result["predicted_summary"], references=result["summary"], rouge_types=["rouge2"])["rouge2"].mid)

with open("output_score.txt", "a") as f:
    print("Result:", rouge.compute(predictions=result["predicted_summary"], references=result["summary"], rouge_types=["rouge2"])["rouge2"].mid, file=f)

pd.DataFrame(result).to_csv("result.csv")