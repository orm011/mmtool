from PIL import Image
import glob
import ray
import ray.data
import json
import numpy as np
import pandas as pd

''' preprocessing scicap folder for a more training friendly format
with one column 'bytes' holding the png bytes of a figure, and the caption column for captions
'''

def captions_to_parquet(root_dir, output_dir):
    json_paths = glob.glob(f'SciCap-Caption-All/**/*.json', root_dir=root_dir, recursive=True)
    paths = sorted(json_paths)
    ds = ray.data.read_json([root_dir + path for path in paths])
    (ds.repartition(num_blocks=32)
        .write_parquet(f'{output_dir}/SciCap-Caption-All.parquet'))
    
def images_to_parquet(root_dir, output_root):
    def process_batch(exb):
        exb1 = exb.assign(path=exb.path.map(lambda x : x[len(root_dir):]))
        exb2 = exb1.assign(**exb1.apply(lambda x : x.path.split('/'), axis=1, result_type='expand').rename({0:'folder', 1:'split', 2:'name'},axis=1))
        return exb2
    
    impaths = glob.glob(f'**/*png', root_dir=root_dir, recursive=True)
    dspng = ray.data.read_binary_files([ root_dir + path for path in impaths ], include_paths=True)
    (dspng.map_batches(process_batch, batch_format='pandas', batch_size=1000)
            .repartition(100)
            .write_parquet(f'{output_root}/SciCap-All-Img.parquet'))
    
def random_paper_perm(merged_df):
    dtp = merged_df.paper_id_noversion
    random_paper_perm = np.random.permutation(dtp.dtype.categories)
    paper_perm_map = dict(zip(random_paper_perm, range(len(random_paper_perm))))
    random_paper_perm_idx = merged_df.paper_id_noversion.map(lambda p : paper_perm_map[p])
    return random_paper_perm_idx


def merge_datasets(data_dir, output_dir):
    # example data dir /home/gridsan/omoll/askem_shared/datasets/scicap_data
    from modin import pandas as pd

    dfcap = pd.read_parquet(f'{data_dir}/SciCap-Caption-All.parquet')
    dfcap_sorted = dfcap.sort_values('figure-ID')
    dfimgs_sorted = dfimgs.sort_values('name')
    dfimgs = pd.read_parquet(f'{data_dir}/SciCap-All-Img.parquet')
    merged_df = dfcap_sorted.merge(dfimgs_sorted, left_on='figure-ID', right_on='name', how='left')
    
    id_noversion = merged_df['paper-ID'].map(lambda x : x.split('v')[0])
    paper_ids = pd.Categorical(id_noversion)
    merged_df = merged_df.assign(paper_id_noversion=paper_ids)
    
    
    random_paper_perm_idx = random_paper_perm(merged_df)
    merged_df = merged_df.assign(random_paper_perm_idx=random_paper_perm_idx)   
    
    random_perm_idx = np.random.permutation(merged_df.shape[0])
    merged_df = merged_df.assign(random_perm_idx=random_perm_idx)
    
    merged_df_shuffled = merged_df.sort_values(['random_paper_perm_idx', 'random_perm_idx'])
    merged_df_shuffled.to_parquet(f'{output_dir}/SciCap-All-Merged-Shuffled.parquet' )