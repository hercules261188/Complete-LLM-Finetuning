AXOLOTL – MINIMUM WORKING STEPS (CLEAN LIST)

STEP 1: Windows → WSL

Wsl



STEP 2: GPU check inside WSL

nvidia-smi

GPU dikhe = OK



STEP 3: Docker + GPU test

docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

GPU dikhe = Docker GPU ready



STEP 4: Start Axolotl Docker

docker run --gpus '"all"' --rm -it axolotlai/axolotl:main-latest

Prompt banega:

root@container-id:/workspace/axolotl#



STEP 5: Fetch example configs

axolotl fetch examples



STEP 6: Run first training (SAFE \& FAST)

axolotl train examples/llama-3/lora-1b.yml

Time: ~15–30 min (GPU pe depend)



STEP 7: Inference (test model)



axolotl inference examples/llama-3/lora-1b.yml \\

&nbsp; --lora-model-dir ./outputs/lora-out

(Optional UI)



axolotl inference examples/llama-3/lora-1b.yml \\

&nbsp; --lora-model-dir ./outputs/lora-out \\

&nbsp; --gradio



STEP 8: (Optional) Merge LoRA

axolotl merge-lora examples/llama-3/lora-1b.yml \\

&nbsp; --lora-model-dir ./outputs/lora-out

Final model:



outputs/lora-out/merged/



