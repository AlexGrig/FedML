import datasets
from pathlib import Path

def load_splitted_part(part_path, split_feature_prefix='image', new_name_split_feature=None):
    part_path = Path(part_path)
    
    part_ds = datasets.load_from_disk(part_path)
    
    #import pdb; pdb.set_trace()
    
    if new_name_split_feature is not None:
        for kk,ds in part_ds.items():
            rename_feature = [str(ff) for ff in list(ds.features) if split_feature_prefix in str(ff)][0]
            
            part_ds[kk] = ds.rename_column(rename_feature, new_name_split_feature)
            
    return part_ds