# DB-GPT-Hub: Text-to-SQL parsing with LLMs

## Baseline

- update time: 2023/12/08
- metric: execution accuracy (ex)
- more details refer to [docs/eval-llm-result.md](https://github.com/anonymity-360/DB-GPT-Hub/blob/main/docs/eval_llm_result.md)

<table style="text-align: center;">
  <tr>
    <th style="text-align: center;">Model</th>
    <th>Method</th>
    <th>Easy</th>
    <th>Medium</th>
    <th>Hard</th>
    <th>Extra</th>
    <th>All</th>
  </tr>
  <tr >
    <td></td>
    <td>base</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
  </tr>
  <tr>
    <td>Llama2-7B-Chat</td>
    <td>lora</td>
    <td>0.887</td>
    <td>0.641</td>
    <td>0.489</td>
    <td>0.331</td>
    <td>0.626</td>
  </tr>
  <tr>
    <td></td>
    <td>qlora</td>
    <td>0.847</td>
    <td>0.623</td>
    <td>0.466</td>
    <td>0.361</td>
    <td>0.608</td>
  </tr>
  <tr>
    <td></td>
    <td>base</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
  </tr>
  <tr>
    <td>Llama2-13B-Chat</td>
    <td>lora</td>
    <td>0.907</td>
    <td>0.729</td>
    <td>0.552</td>
    <td>0.343</td>
    <td>0.68</td>
  </tr>
  <tr>
    <td></td>
    <td>qlora</td>
    <td>0.911</td>
    <td>0.7</td>
    <td>0.552</td>
    <td>0.319</td>
    <td>0.664</td>
  </tr>
  <tr>
    <td></td>
    <td>base</td>
    <td>0.214</td>
    <td>0.177</td>
    <td>0.092</td>
    <td>0.036</td>
    <td>0.149</td>
  </tr>
  <tr>
  <td>CodeLlama-7B-Instruct</td>
    <td>lora</td>
    <td>0.923</td>
    <td>0.756</td>
    <td>0.586</td>
    <td>0.349</td>
    <td>0.702</td>
  </tr>
  <tr>
    <td></td>
    <td>qlora</td>
    <td>0.911</td>
    <td>0.751</td>
    <td>0.598</td>
    <td>0.331</td>
    <td>0.696</td>
  </tr>
  <tr>
    <td></td>
    <td>base</td>
    <td>0.698</td>
    <td>0.601</td>
    <td>0.408</td>
    <td>0.271</td>
    <td>0.539</td>
  </tr>
  <tr>
    <td>CodeLlama-13B-Instruct</td>
    <td>lora</td>
    <td>0.94</td>
    <td>0.789</td>
    <td>0.684</td>
    <td>0.404</td>
    <td>0.746</td>
  </tr>
  <tr>
    <td></td>
    <td>qlora</td>
    <td>0.94</td>
    <td>0.774</td>
    <td>0.626</td>
    <td>0.392</td>
    <td>0.727</td>
  </tr>
  <tr>
    <td></td>
    <td>base</td>
    <td>0.577</td>
    <td>0.352</td>
    <td>0.201</td>
    <td>0.066</td>
    <td>0.335</td>
  </tr>
  <tr>
    <td>Baichuan2-7B-Chat</td>
    <td>lora</td>
    <td>0.871</td>
    <td>0.63</td>
    <td>0.448</td>
    <td>0.295</td>
    <td>0.603</td>
  </tr>
  <tr>
  <td></td>
    <td>qlora</td>
    <td>0.891</td>
    <td>0.637</td>
    <td>0.489</td>
    <td>0.331</td>
    <td>0.624</td>
  </tr>
  <tr>
    <td></td>
    <td>base</td>
    <td>0.581</td>
    <td>0.413</td>
    <td>0.264</td>
    <td>0.187</td>
    <td>0.392</td>
  </tr>
    <tr>
    <td>Baichuan2-13B-Chat</td>
    <td>lora</td>
    <td>0.903</td>
    <td>0.702</td>
    <td>0.569</td>
    <td>0.392</td>
    <td>0.678</td>
    </tr>
  <tr>
  <td></td>
    <td>qlora</td>
    <td>0.895</td>
    <td>0.675</td>
    <td>0.58</td>
    <td>0.343</td>
    <td>0.659</td>
  </tr>
  <tr>
  <td></td>
  <td>base</td>
  <td>0.395</td>
  <td>0.256</td>
  <td>0.138</td>
  <td>0.042</td>
  <td>0.235</td>
  </tr>
<tr>
<td>Qwen-7B-Chat</td>
  <td>lora</td>
  <td>0.855</td>
  <td>0.688</td>
  <td>0.575</td>
  <td>0.331</td>
  <td>0.652</td>
  </tr>
  <tr>
    <td></td>
    <td>qlora</td>
    <td>0.911</td>
    <td>0.675</td>
    <td>0.575</td>
    <td>0.343</td>
    <td>0.662</td>
  </tr>
  <tr>
  <td></td>
  <td>base</td>
  <td>0.871</td>
  <td>0.632</td>
  <td>0.368</td>
  <td>0.181</td>
  <td>0.573</td>
  </tr>
  <tr>
    <td>Qwen-14B-Chat</td>
    <td>lora</td>
    <td>0.895</td>
    <td>0.702</td>
    <td>0.552</td>
    <td>0.331</td>
    <td>0.663</td>
  </tr>
  <tr>
    <td></td>
    <td>qlora</td>
    <td>0.919</td>
    <td>0.744</td>
    <td>0.598</td>
    <td>0.367</td>
  <td>0.701</td>
  </tr>
    <tr>
    <td></td>
    <td>base</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
  </tr>
  <tr>
    <td>ChatGLM3-6b</td>
    <td>lora</td>
    <td>0.855</td>
    <td>0.605</td>
    <td>0.477</td>
    <td>0.271</td>
    <td>0.59</td>
  </tr>
  <tr>
    <td></td>
    <td>qlora</td>
    <td>0.843</td>
    <td>0.603</td>
    <td>0.506</td>
    <td>0.211</td>
    <td>0.581</td>
  </tr>
</table>

## Contents

- [DB-GPT-Hub: Text-to-SQL parsing with LLMs](#db-gpt-hub-text-to-sql-parsing-with-llms)
  - [Baseline](#baseline)
  - [Contents](#contents)
  - [1. What is DB-GPT-Hub](#1-what-is-db-gpt-hub)
  - [2. Fine-tuning Text-to-SQL](#2-fine-tuning-text-to-sql)
    - [2.1. Dataset](#21-dataset)
    - [2.2. Model](#22-model)
  - [3. Usage](#3-usage)
    - [3.1. Environment preparation](#31-environment-preparation)
    - [3.2 Quick Start](#32-quick-start)
    - [3.3. Data preparation](#33-data-preparation)
    - [3.4. Model fine-tuning](#34-model-fine-tuning)
    - [3.5. Model Predict](#35-model-predict)
    - [3.6 Model Weights](#36-model-weights)
      - [3.6.1 Model and fine-tuned weight merging](#361-model-and-fine-tuned-weight-merging)
    - [3.7 Model Evaluation](#37-model-evaluation)

## 1. What is DB-GPT-Hub

DB-GPT-Hub is an experimental project that leverages Large Language Models (LLMs) to achieve Text-to-SQL parsing. The project encompasses various stages, including data collection, data preprocessing, model selection and construction, and fine-tuning of model weights. Through these processes, our aim is to enhance Text-to-SQL capabilities while reducing model training costs, thus enabling more developers to contribute to improving Text-to-SQL accuracy. Our ultimate goal is to realize automated question-answering capabilities based on databases, allowing users to execute complex database queries using natural language descriptions.

To date, we have successfully integrated multiple large models and established a comprehensive workflow that includes data processing, Supervised Fine-Tuning (SFT) model training, prediction output, and evaluation. The code developed for this project is easily reusable within the project itself.

As of October 10, 2023, we have used this project to fine-tune the open-source 13B-sized model, incorporating more relevant data. Under zero-shot prompts and utilizing [the Spider-based test-suite](https://github.com/taoyds/test-suite-sql-eval), we have achieved an execution accuracy rate of 0.764 for a database with a size of 1.27G. Additionally, the execution accuracy for the database pointed to by [the Spider official website](https://yale-lily.github.io/spider), with a size of 95M, stands at 0.825.

## 2. Fine-tuning Text-to-SQL

We enhance the Text-to-SQL performance by applying Supervised Fine-Tuning (SFT) on large language models.

### 2.1. Dataset

The primary dataset for this project's examples is the **Spider** dataset:

- [SPIDER](https://yale-lily.github.io/spider): A complex text2sql dataset across domains, containing 10,181 natural language queries, 5,693 SQL distributed across 200 separate databases, covering 138 different domains.[download link](https://drive.google.com/uc?export=download&id=1TqleXec_OykOYFREKKtschzY29dUcVAQ)

Other text2sql datasets available:

- [WikiSQL:](https://github.com/salesforce/WikiSQL) A large semantic parsing dataset consisting of 80,654 natural statement expressions and sql annotations of 24,241 tables. Each query in WikiSQL is limited to the same table and does not contain complex operations such as sorting, grouping The queries in WikiSQL are limited to the same table and do not include complex operations such as sorting, grouping, subqueries, etc.
- [CHASE](https://xjtu-intsoft.github.io/chase/): A cross-domain multi-round interactive text2sql Chinese dataset containing a list of 5,459 multi-round questions consisting of 17,940 <query, SQL> binary groups across 280 different domain databases.
- [BIRD-SQL:](https://bird-bench.github.io/) A large-scale cross-domain text-to-SQL benchmark in English, with a particular focus on large database content. The dataset contains 12,751 text-to-SQL data pairs and 95 databases with a total size of 33.4 GB across 37 occupational domains. The BIRD-SQL dataset bridges the gap between text-to-SQL research and real-world applications by exploring three additional challenges, namely dealing with large and messy database values, external knowledge inference and optimising SQL execution efficiency.
- [CoSQL:](https://yale-lily.github.io/cosql) A corpus for building cross-domain conversational text-to-SQL systems. It is a conversational version of the Spider and SParC tasks. CoSQL consists of 30k+ rounds and 10k+ annotated SQL queries from Wizard-of-Oz's collection of 3k conversations querying 200 complex databases across 138 domains. Each conversation simulates a realistic DB query scenario in which a staff member explores the database as a user and a SQL expert uses SQL to retrieve answers, clarify ambiguous questions, or otherwise inform.
- Following the processing template of [NSQL](https://github.com/NumbersStationAI/NSQL), the dataset underwent basic processing, yielding approximately [20W dataset](https://huggingface.co/datasets/Healthy13/Text2SQL/tree/main)

### 2.2. Model

DB-GPT-Hub currently supports the following base models:

- [X] CodeLlama
- [X] Baichuan2
- [X] LLaMa/LLaMa2
- [X] Falcon
- [X] Qwen
- [X] XVERSE
- [X] ChatGLM2
- [X] ChatGLM3
- [X] internlm
- [X] sqlcoder-7b(mistral)
- [X] sqlcoder2-15b(starcoder)

The model is fine-tuned based on a quantization bit of 4 using Quantized Learning over Redundant Architecture (QLoRA). The minimum hardware requirements for this can be referred to as follows:

| Model Parameters | GPU RAM | CPU RAM | DISK   |
| ---------------- | ------- | ------- | ------ |
| 7b               | 6GB     | 3.6GB   | 36.4GB |
| 13b              | 13.4GB  | 5.9GB   | 60.2GB |

All the related parameters are set to the minimum, with a batch size of 1 and max length of 512. Based on experience, for better performance, it is recommended to set the related length values to 1024 or 2048.

## 3. Usage

### 3.1. Environment preparation

```
git clone https://github.com/anonymity-360/DB-GPT-Hub.git
cd DB-GPT-Hub
conda create -n dbgpt_hub python=3.10 
conda activate dbgpt_hub
pip install poetry
poetry install
```

### 3.2 Quick Start

Firstly, install `dbgpt-hub` with the following command

`pip install dbgpt-hub`

Then, set up the arguments and run the whole process.

```python
from dbgpt_hub.data_process import preprocess_sft_data
from dbgpt_hub.train import start_sft
from dbgpt_hub.predict import start_predict
from dbgpt_hub.eval import start_evaluate

# Config the input datasets
data_folder = "dbgpt_hub/data"
data_info = [
        {
            "data_source": "spider",
            "train_file": ["train_spider.json", "train_others.json"],
            "dev_file": ["dev.json"],
            "tables_file": "tables.json",
            "db_id_name": "db_id",
            "is_multiple_turn": False,
            "train_output": "spider_train.json",
            "dev_output": "spider_dev.json",
        }
]

# Config training parameters
train_args = {
            "model_name_or_path": "codellama/CodeLlama-13b-Instruct-hf",
            "do_train": True,
            "dataset": "example_text2sql_train",
            "max_source_length": 2048,
            "max_target_length": 512,
            "finetuning_type": "lora",
            "lora_target": "q_proj,v_proj",
            "template": "llama2",
            "lora_rank": 64,
            "lora_alpha": 32,
            "output_dir": "dbgpt_hub/output/adapter/CodeLlama-13b-sql-lora",
            "overwrite_cache": True,
            "overwrite_output_dir": True,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "lr_scheduler_type": "cosine_with_restarts",
            "logging_steps": 50,
            "save_steps": 2000,
            "learning_rate": 2e-4,
            "num_train_epochs": 8,
            "plot_loss": True,
            "bf16": True,
}

# Config predict parameters
predict_args = {
            "model_name_or_path": "codellama/CodeLlama-13b-Instruct-hf",
            "template": "llama2",
            "finetuning_type": "lora",
            "checkpoint_dir": "dbgpt_hub/output/adapter/CodeLlama-13b-sql-lora",
            "predict_file_path": "dbgpt_hub/data/eval_data/dev_sql.json",
            "predict_out_dir": "dbgpt_hub/output/",
            "predicted_out_filename": "pred_sql.sql",
}

# Config evaluation parameters
evaluate_args =  {
            "input": "./dbgpt_hub/output/pred/pred_sql_dev_skeleton.sql",
            "gold": "./dbgpt_hub/data/eval_data/gold.txt",
            "gold_natsql": "./dbgpt_hub/data/eval_data/gold_natsql2sql.txt",
            "db": "./dbgpt_hub/data/spider/database",
            "table": "./dbgpt_hub/data/eval_data/tables.json",
            "table_natsql": "./dbgpt_hub/data/eval_data/tables_for_natsql2sql.json",
            "etype": "exec",
            "plug_value": True,
            "keep_distict": False,
            "progress_bar_for_each_datapoint": False,
            "natsql": False,
}

# Run the whole fine-tuning workflow
preprocess_sft_data(
      data_folder = data_folder,
      data_info = data_info
)

start_sft(train_args)
start_predict(predict_args)
start_evaluate(evaluate_args)
```

### 3.3. Data preparation

DB-GPT-Hub uses the information matching generation method for data preparation, i.e. the SQL + Repository generation method that combines table information. This method combines data table information to better understand the structure and relationships of the data table, and is suitable for generating SQL statements that meet the requirements.

Download the [Spider dataset](https://drive.google.com/uc?export=download&id=1TqleXec_OykOYFREKKtschzY29dUcVAQ)) from the Spider dataset link. By default, after downloading and extracting the data, place it in the dbgpt_hub/data directory, i.e., the path should be `dbgpt_hub/data/spider`.

For the data preprocessing part, simply **run the following script** :

```bash
## generate train and dev(eval) data
poetry run sh dbgpt_hub/scripts/gen_train_eval_data.sh
```

In the directory `dbgpt_hub/data/`, you will find the newly generated training file example_text2sql_train.json and testing file example_text2sql_dev.json, containing 8659 and 1034 entries respectively. For the data used in subsequent fine-tuning, set the parameter `file_name` value to the file name of the training set in dbgpt_hub/data/dataset_info.json, such as example_text2sql_train.json

The data in the generated JSON looks something like this:

```
    {
        "db_id": "department_management",
        "instruction": "I want you to act as a SQL terminal in front of an example database, you need only to return the sql command to me.Below is an instruction that describes a task, Write a response that appropriately completes the request.\n\"\n##Instruction:\ndepartment_management contains tables such as department, head, management. Table department has columns such as Department_ID, Name, Creation, Ranking, Budget_in_Billions, Num_Employees. Department_ID is the primary key.\nTable head has columns such as head_ID, name, born_state, age. head_ID is the primary key.\nTable management has columns such as department_ID, head_ID, temporary_acting. department_ID is the primary key.\nThe head_ID of management is the foreign key of head_ID of head.\nThe department_ID of management is the foreign key of Department_ID of department.\n\n",
        "input": "###Input:\nHow many heads of the departments are older than 56 ?\n\n###Response:",
        "output": "SELECT count(*) FROM head WHERE age  >  56",
        "history": []
    }, 
```

The data processing code of `chase`, `cosql` and `sparc` has been embedded in the data processing code of the project. After downloading the data set according to the above link, you only need to add `in`dbgpt_hub/configs/config.py ` Just loosen the corresponding code comment in SQL_DATA_INFO`.

### 3.4. Model fine-tuning

The model fine-tuning supports both LoRA and QLoRA methods. We can run the following command to fine-tune the model. By default, with the parameter --quantization_bit, it uses the QLoRA fine-tuning method. To switch to LoRAs, simply remove the related parameter from the script.
Run the command:

```bash
poetry run sh dbgpt_hub/scripts/train_sft.sh
```

After fine-tuning, the model weights will be saved by default in the adapter folder, specifically in the dbgpt_hub/output/adapter directory.

If you're using **multi-GPU training and want to utilize deepseed**, you should modify the default content in train_sft.sh. The change  is:

```
CUDA_VISIBLE_DEVICES=0 python dbgpt_hub/train/sft_train.py \
    --quantization_bit 4 \
    ...
```

change to ：

```
deepspeed --num_gpus 2  dbgpt_hub/train/sft_train.py \
    --deepspeed dbgpt_hub/configs/ds_config.json \
    --quantization_bit 4 \
    ...
```

if you need  order card  id

```
deepspeed --include localhost:0,1  dbgpt_hub/train/sft_train.py \
    --deepspeed dbgpt_hub/configs/ds_config.json \
    --quantization_bit 4 \
    ...
```

The other parts that are omitted (…) can be kept consistent. If you want to change the default deepseed configuration, go into the `dbgpt_hub/configs` directory and make changes to ds_config.json as needed,the default is stage2.

In the script, during fine-tuning, different models correspond to key parameters lora_target and template, as shown in the following table:

| model name                                            | lora_target     | template  |
| ----------------------------------------------------- | --------------- | --------- |
| [LLaMA-2](https://huggingface.co/meta-llama)             | q_proj,v_proj   | llama2    |
| [CodeLlama-2](https://huggingface.co/codellama/)         | q_proj,v_proj   | llama2    |
| [Baichuan2](https://github.com/baichuan-inc/Baichuan2)   | W_pack          | baichuan2 |
| [Qwen](https://github.com/QwenLM/Qwen-7B)                | c_attn          | chatml    |
| [sqlcoder-7b](https://huggingface.co/defog/sqlcoder-7b)  | q_proj,v_proj   | mistral   |
| [sqlcoder2-15b](https://huggingface.co/defog/sqlcoder2)  | c_attn          | default   |
| [InternLM](https://github.com/InternLM/InternLM)         | q_proj,v_proj   | intern    |
| [XVERSE](https://github.com/xverse-ai/XVERSE-13B)        | q_proj,v_proj   | xverse    |
| [ChatGLM2](https://github.com/THUDM/ChatGLM2-6B)         | query_key_value | chatglm2  |
| [LLaMA](https://github.com/facebookresearch/llama)       | q_proj,v_proj   | -         |
| [BLOOM](https://huggingface.co/bigscience/bloom)         | query_key_value | -         |
| [BLOOMZ](https://huggingface.co/bigscience/bloomz)       | query_key_value | -         |
| [Baichuan](https://github.com/baichuan-inc/baichuan-13B) | W_pack          | baichuan  |
| [Falcon](https://huggingface.co/tiiuae/falcon-7b)        | query_key_value | -         |

 In `train_sft.sh` , other key parameters are as follows:

> quantization_bit: Indicates whether quantization is applied, with valid values being [4 or 8].
> model_name_or_path: The path of the LLM (Large Language Model).
> dataset: Specifies the name of the training dataset configuration, corresponding to the outer key value in dbgpt_hub/data/dataset_info.json, such as example_text2sql.
> max_source_length: The length of the text input into the model. If computing resources allow, it can be set as large as possible, like 1024 or 2048.
> max_target_length: The length of the SQL content output by the model; 512 is generally sufficient.
> output_dir: The output path of the Peft module during SFT (Supervised Fine-Tuning), set by default to `dbgpt_hub/output/adapter/` .
> per_device_train_batch_size: The size of the batch. If computing resources allow, it can be set larger; the default is 1.
> gradient_accumulation_steps: The number of steps for accumulating gradients before an update.
> save_steps: The number of steps at which model checkpoints are saved; it can be set to 100 by default.
> num_train_epochs: The number of epochs for training the dataset.

### 3.5. Model Predict

Under the project directory ./dbgpt_hub/output/pred/, this folder is the default output location for model predictions(if not exist, just mkdir).

```bash
poetry run sh ./dbgpt_hub/scripts/predict_sft.sh
```

In the script, by default with the parameter `--quantization_bit`, it predicts using QLoRA. Removing it switches to the LoRA prediction method.
The value of the parameter `predicted_input_filename`  is your predict test dataset file.  `--predicted_out_filename` is the file name of the model's predicted results.

### 3.6 Model Weights

You can find the second corresponding model weights  from Huggingface [hg-eosphoros-ai
](https://huggingface.co/Wangzaistone123/CodeLlama-13b-sql-lora)  ,we uploaded the LoRA weights in October,which execution accuracy on the Spider evaluation set reached 0.789.

#### 3.6.1 Model and fine-tuned weight merging

If you need to merge the weights of the trained base model and the fine-tuned Peft module to export a complete model, execute the following model export script:

```bash
poetry run sh ./dbgpt_hub/scripts/export_merge.sh
```

Be sure to replace the parameter path values in the script with the paths corresponding to your project.

### 3.7 Model Evaluation

To evaluate model performance on the dataset, default is spider dev dataset.
Run the following command:

```bash
poetry run python dbgpt_hub/eval/evaluation.py --plug_value --input Your_model_pred_file
```

You can find the results of our latest review and part of experiment results [here](docs/eval_llm_result.md)
**Note**: The database pointed to by the default code is a 95M database downloaded from [Spider official website] (https://yale-lily.github.io/spider). If you need to use Spider database (size 1.27G) in [test-suite](https://github.com/taoyds/test-suite-sql-eval), please download the database in the link to the custom directory first, and run the above evaluation command which add parameters and values like `--db Your_download_db_path`.
