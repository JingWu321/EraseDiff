from T2IBenchmark import calculate_fid
from T2IBenchmark.datasets import get_coco_fid_stats

fid, _ = calculate_fid(
    './evaluation_folder/coco/SD',
    get_coco_fid_stats()
)

print(f'FID: {fid}')


from T2IBenchmark import calculate_clip_score
import pandas as pd

root = './evaluation_folder/coco/SD/'

df = pd.read_csv('./prompts/coco_30k.csv')
captions_mapping = {}
img_paths = []
for i, row in df.iterrows():
    case_number = row.case_number
    if case_number < 100:
        img_path = root + str(case_number) + '_0.png'
        img_paths.append(img_path)
        captions_mapping[img_path] = str(row.prompt)

clip_score = calculate_clip_score(img_paths, captions_mapping=captions_mapping, batch_size=16, dataloader_workers=4)


# CUDA_VISIBLE_DEVICES=1 python test.py


