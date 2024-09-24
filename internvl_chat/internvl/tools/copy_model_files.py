import os
from os.path import join, isfile

root_folder = join(os.path.dirname(__file__), '..')


source_dir = join(root_folder, 'ark_lvlm', 'model')

target_dirs = [
    'models',
    'pretrained'
]


for target_dir in target_dirs:
    target_subdirs = [join(root_folder, target_dir, dir) for dir in os.listdir(join(root_folder, target_dir))]
    for target_subdir in target_subdirs:
        if isfile(target_dir) or target_subdir.endswith('.gitkeep'):
            continue

        for file_name in os.listdir(source_dir):
            source_file = join(source_dir, file_name)
            if not os.path.isfile(source_file):
                continue

            target_file = join(target_subdir, file_name)
            try:
                if os.path.exists(target_file):
                    os.remove(target_file)
                os.symlink(source_file, target_file)
                print(f"Link created for {target_file}")
            except Exception as e:
                print(e)
