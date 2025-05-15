import os
import zipfile
import subprocess
import sys
import requests
import json
import random

def download_and_unzip(url: str, destination: str):
    if not os.path.exists(destination):
        os.makedirs(destination)
    filename = url.split('/')[-1]
    file_path = os.path.join(destination, filename)

    if os.path.exists(file_path):
        print(f'{filename} already exists. Skipping download.')
        return

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    gb = 1024*1024*1024

    chunk_flag = total_size / gb > 1
    chunk_size = int(total_size/100) if chunk_flag else total_size

    with open(file_path, 'wb') as file:
        for idx, chunk in enumerate(response.iter_content(chunk_size=chunk_size)):
            if not chunk: continue
            if chunk_flag:
                print(f"{idx}% downloading: "
                      f"{round((idx*chunk_size)/gb,1)}GB / {round(total_size/gb,1)}GB")
            file.write(chunk)
    print(f"'{filename}' downloaded successfully.")

    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(destination)
    print(f"'{filename}' extracted successfully.")
    os.remove(file_path)

def compile_nms():
    """Compile the C++ NMS implementation"""
    print("Compiling C++ NMS implementation...")
    try:
        subprocess.run(
            [sys.executable, "setup.py", "install", "--user"],
            cwd="../env/libs/utils",
            check=True
        )
        print("NMS compilation successful!")
    except subprocess.CalledProcessError as e:
        print(f"Error compiling NMS: {e}")
        sys.exit(1)

# First compile NMS
compile_nms()

data_path = '../env/data/pt'
os.makedirs(data_path, exist_ok=True)

# Training data
train_annot_url = 'https://storage.googleapis.com/dm-perception-test/zip_data/challenge_sound_localisation_train_annotations.zip'
download_and_unzip(train_annot_url, data_path)
train_video_feat_url = 'https://storage.googleapis.com/dm-perception-test/zip_data/action_localisation_train_video_features.zip'
download_and_unzip(train_video_feat_url, data_path)
train_audio_feat_url = 'https://storage.googleapis.com/dm-perception-test/zip_data/sound_localisation_train_audio_features.zip'
download_and_unzip(train_audio_feat_url, data_path)

# — split training data into 80% train / 20% valid —
json_train_file = os.path.join(data_path, 'challenge_sound_localisation_train.json')
with open(json_train_file, 'r') as f:
    json_db = json.load(f)

all_ids = list(json_db.keys())
random.shuffle(all_ids)
split_idx = int(len(all_ids) * 0.8)
train_ids = all_ids[:split_idx]
valid_ids = all_ids[split_idx:]

# build new dicts, updating each entry's split tag
train_db = {}
valid_db = {}
for vid in train_ids:
    entry = json_db[vid]
    entry['metadata']['split'] = 'train'
    train_db[vid] = entry
for vid in valid_ids:
    entry = json_db[vid]
    entry['metadata']['split'] = 'valid'
    valid_db[vid] = entry

# overwrite train JSON and write out valid JSON
with open(json_train_file, 'w') as f:
    json.dump(train_db, f)
json_valid_file = os.path.join(data_path, 'challenge_sound_localisation_valid.json')
with open(json_valid_file, 'w') as f:
    json.dump(valid_db, f)

print(f"Split {len(train_ids)} videos for training and {len(valid_ids)} for validation.")

# now split the .npy feature files
video_train_dir = os.path.join(data_path, 'action_localisation_train_video_features')
audio_train_dir = os.path.join(data_path, 'sound_localisation_train_audio_features')

video_valid_dir = os.path.join(data_path, 'action_localisation_valid_video_features')
os.makedirs(video_valid_dir, exist_ok=True)
audio_valid_dir = os.path.join(data_path, 'sound_localisation_valid_audio_features')
os.makedirs(audio_valid_dir, exist_ok=True)

for vid in valid_ids:
    # move video features
    src_vid = os.path.join(video_train_dir, vid + '.npy')
    dst_vid = os.path.join(video_valid_dir, vid + '.npy')
    if os.path.exists(src_vid):
        os.rename(src_vid, dst_vid)

    # move audio features
    src_aud = os.path.join(audio_train_dir, vid + '.npy')
    dst_aud = os.path.join(audio_valid_dir, vid + '.npy')
    if os.path.exists(src_aud):
        os.rename(src_aud, dst_aud)

print("Feature directories split for train/valid.")

# — now: treat validation as our test set —
test_path = 'test_data/pt'
os.makedirs(test_path, exist_ok=True)

# use the same valid_* URLs, but download into test_path
valid_annot_url     = 'https://storage.googleapis.com/dm-perception-test/zip_data/challenge_sound_localisation_valid_annotations.zip'
valid_video_feat_url = 'https://storage.googleapis.com/dm-perception-test/zip_data/action_localisation_valid_video_features.zip'
valid_audio_feat_url = 'https://storage.googleapis.com/dm-perception-test/zip_data/sound_localisation_valid_audio_features.zip'

download_and_unzip(valid_annot_url,     test_path)
download_and_unzip(valid_video_feat_url, test_path)
download_and_unzip(valid_audio_feat_url, test_path)

# rename anything with "_valid_" to "_test_"
for old_name, new_name in [
    ('challenge_sound_localisation_valid.json',        'challenge_sound_localisation_test.json'),
    ('action_localisation_valid_video_features',       'action_localisation_test_video_features'),
    ('sound_localisation_valid_audio_features',        'sound_localisation_test_audio_features'),
]:
    src = os.path.join(test_path, old_name)
    dst = os.path.join(test_path, new_name)
    if os.path.exists(src):
        os.rename(src, dst)

print("All preparation steps completed successfully!")
with open("prepared", 'w') as writer:
    pass
