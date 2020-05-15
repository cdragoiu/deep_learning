import os, random, shutil

def train_test_split(path, ratio):
    '''
    Split files into train and test based on the split ratio. The selection is randomized.
    Args:
        path:  location of files to process (subfolders are considered image classes)
        ratio: percentage of files to be used for training
    '''

    abs_path = os.path.abspath(path.rstrip('/'))
    dir_names = ['train', 'test']

    # reset folder structure if present
    for dir in dir_names:
        test_dir = abs_path + '/' + dir
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

    # process subfolders
    for class_name in os.listdir(abs_path):

        # skip hidden folders
        if class_name.startswith('.'):
            continue

        # create folder structure
        try:
            for dir in dir_names:
                os.makedirs(abs_path + '/' + dir + '/' + class_name)
        except OSError:
            print('error: unable to create folder structure')

        # randomize file selection
        split_files = {}
        file_names = os.listdir(abs_path + '/' + class_name)
        split_files[dir_names[0]] = random.sample(file_names, int(ratio * len(file_names)))
        split_files[dir_names[1]] = [f for f in file_names if f not in split_files[dir_names[0]]]

        # process files
        for dir in dir_names:
            for name in split_files[dir]:
                test_name = abs_path + '/' + class_name + '/' + name

                # skip empty files
                if os.path.getsize(test_name) == 0:
                    print('warning: file "' + name + '" is empty')
                    continue

                os.symlink(test_name, test_name.replace(abs_path, abs_path + '/' + dir))
