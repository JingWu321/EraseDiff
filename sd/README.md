# EraseDiff for Stable Diffusion
This is the official repository for EraseDiff for diffusion models. The code structure of this project is adapted from the [ESD](https://github.com/rohitgandikota/erasing/tree/main), [SA](https://github.com/clear-nus/selective-amnesia/tree/main/sd) and [SalUn](https://github.com/OPTML-Group/Unlearn-Saliency/tree/master/SD) codebase.

# Requirements
Install the requirements using a `conda` environment:
```
conda env create -f environment.yml
```

## Download SD v1.4 Checkpoint
We will use the SD v1.4 checkpoint (with EMA). You can either download it from the official HuggingFace [link](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original) and move it to the root directory of this project, or alternatively
```
wget https://huggingface.co/CompVis/stable-diffusion-v-1-4-original
```

# Forgetting Training with EraseDiff
We consider forgetting the nudity in the following steps.

1. Generate Samples for training.

```
python eval_scripts/generate-images_nude.py --prompts_path 'prompts/nsfw.csv' --save_path 'evaluation_folder/NSFW&NotNSFW/' --model_name 'SD' --device 'cuda:0' --num_samples $N
```

2. Forgetting Training with SA.

```
python erasediff_nsfw.py --train_method 'noxattn' --device '2' --epochs $E --K_steps $K --lambda_bome $lambda --lr 1e-5 --batch_size 16
```


# Evaluation

## NudeNet

1. Generate the images from the I2P prompts.
   
```
python eval_scripts/generate-images_i2p.py --prompts_path 'prompts/unsafe-prompts4703.csv' --save_path 'evaluation_folder/i2p/' --model_name checkpoint --device 'cuda:2' --num_samples 1
```

2. Run the NudeNet evaluation.

```
python nudenet_evaluator.py path/to/outdir
```

The statistics of the relevant nudity concepts will be printed to screen like such

```
BUTTOCKS_EXPOSED: 28
FEMALE_BREAST_EXPOSED: 276
FEMALE_GENITALIA_EXPOSED: 13
MALE_BREAST_EXPOSED: 44
ANUS_EXPOSED: 0
FEET_EXPOSED: 55
ARMPITS_EXPOSED: 158
BELLY_EXPOSED: 162
MALE_GENITALIA_EXPOSED: 7
```


