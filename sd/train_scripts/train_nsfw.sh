# # Nudity erasure

## [1] Perform unlearning
python erasediff_nsfw.py --train_method 'noxattn' --device '0' --epochs 5 --K_steps 2 --lambda_bome 1. --lr 1e-5 --batch_size 16

## [2] Generate images (step 160 is enough)
python eval_scripts/generate-images_nude.py --prompts_path 'prompts/nsfw.csv' --save_path 'evaluation_folder/nude/xxx' --model_name 'models/xxx' --device 'cuda:0' --num_samples 10
python eval_scripts/generate-images_i2p.py --prompts_path 'prompts/unsafe-prompts4703.csv' --save_path 'evaluation_folder/i2p/xxx' --model_name 'models/xxx' --device 'cuda:1' --num_samples 1
python eval_scripts/generate-images_coco.py --prompts_path 'prompts/coco_30k.csv' --save_path 'evaluation_folder/coco/xxx' --model_name 'models/xxx' --device 'cuda:2' --num_samples 1

## [3] NudeNet Evaluation
python nudenet_evaluator.py path/to/outdir
