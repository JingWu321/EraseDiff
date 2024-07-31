# # imagenette

## Perform unlearning
# python erasediff_rl.py --class_to_forget 0 --train_method 'full' --device '0' --epochs 2 --K_steps 1 --lr 1e-5 --lambda_bome 1. --batch_size 16


## Generate images (step 60 is enough)
# python eval_scripts/generate-images.py --prompts_path 'prompts/imagenette.csv' --save_path 'evaluation_folder/imagenette/' --model_name './models/xxx' --device 'cuda:1' --num_samples 10


## Evaluation
# python eval_scripts/compute-fid.py --folder_path './evaluation_folder/xxx'
# python eval_scripts/imageclassify.py --prompts_path './prompts/imagenette.csv' --folder_path './evaluation_folder/xxx'

