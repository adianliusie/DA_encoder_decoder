from types import SimpleNamespace
import sys

sys.path.append('/home/alta/Conversational/OET/al826/2022/DA_classification/1-parallel_exp')
from src.eval.integrad_handler import IntegradHandler


eval_path = f"swda/standard/dev.json"

test_args = {'eval_path':eval_path,
             'bsz':1,
             'lim':None}

test_args = SimpleNamespace(**test_args)

E = IntegradHandler('sep_focus_utt/full', hpc=True)

for conv_num in [1,5,6,13,14,21]:
    if E.dir.file_exists(f'integrad/{conv_num}'):
        print(f'skipping {conv_num}')
        continue
    output = E.conv_integrad(test_args, conv_num=conv_num, N=100)
    E.dir.save_dict(f'integrad/{conv_num}', output)
