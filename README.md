# ADL-HW3
## Task: Instruction Tuning (Classical Chinese)

## Environment
```bash
torch==2.4.1, transformers==4.45.1, bitsandbytes==0.44.1, peft==0.13.0
```

## Training
1. Environment Setup
   
   The environment setup can be tricky in Google Colab. You should install the required packages mentioned above and restart the working session for it to take effect.

2. Mount Your Drive
   
   Mount your Google Drive to the ipynb file, make sure that your have an ADL/hw3 directory created in your drive. Training data should be stored in ADL/hw3/data.

3. Real Training Session
   
   Now, the real training session starts. Including importing the packages you need training, validation, and saving the model. There are no arguments to pass so you will have to manually modify the hyperparameters in the code if you want. Execute each code block step by step to train.

## Inference
To run inference using the provided run.sh script, use the following command format:
```bash
bash run.sh \
    /path/to/model-folder \
    /path/to/adapter_checkpoint \
    /path/to/input.json \
    /path/to/output.json
```
Example:
```bash
bash run.sh \
    /home/user/models/'zake7749/gemma-2-2b-it-chinese-kyara-dpo' \
    ./adapter_checkpoint \
    ./data/public_test.json \
    ./output/predictions.json
```
