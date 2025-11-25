import pickle
from pathlib import Path
import sys

def check_pkl(pkl_path):
    print(f"Checking {pkl_path}")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print("Keys:", data.keys())
    if 'segments' in data:
        print(f"Found {len(data['segments'])} segments")
        if len(data['segments']) > 0:
            seg = data['segments'][0]
            print("First segment keys:", seg.keys())
            if 'unique_labels' in seg:
                print("unique_labels:", seg['unique_labels'])
            else:
                print("unique_labels NOT FOUND in segment info")
    else:
        print("segments key NOT FOUND")

if __name__ == "__main__":
    data_dir = Path(r"E:\data\DALES\dales_las\bin\train")
    pkl_files = list(data_dir.glob("*.pkl"))
    if pkl_files:
        check_pkl(pkl_files[0])
    else:
        print(f"No pkl files found in {data_dir}")
