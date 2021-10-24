import os
import pandas as pd
import glob
import nibabel as nib
from nilearn.image import resample_img
import numpy as np
from sklearn.model_selection import StratifiedKFold


# split data train, validate
def generate_csv(options):
    # order ISBI dataset
    train_csv_path = options["train_csv_path"]
    modalities_names = options['preprocess_x_names']
    modalities = options['modalities']
    masks_names = options['preprocess_y_names']
    masks = options['masks']
    # generate csv file of files names and split dataset
    _, dirs, _ = next(os.walk(options['train_folder']))
    if os.path.isfile(train_csv_path):
        os.remove(train_csv_path)

    train_data = pd.DataFrame(columns=['root_path', 'patient_id', 'study', *masks, *modalities, 'fold'])
    train_data = train_data.astype({"study": str})

    for dir_ in dirs:
        patient_id = dir_.split('_')[0]
        study = "_"+dir_.split('_')[1]
        root_path = os.path.join(options['train_folder'], dir_, options['tmp_folder'])
        df = pd.DataFrame([[root_path, patient_id, study, *masks_names, *modalities_names, 1]], columns=['root_path', 'patient_id', 'study', *masks, *modalities, 'fold'])
        train_data = train_data.append(df)
    train_data.reset_index(inplace=True)
    train_data.drop(columns=['index'], inplace=True)
    train_data.to_csv(train_csv_path, index=False)
    return train_data


def resize_images(options):
    # resize images just for test as image large remove it later
    root, dirs, _ = next(os.walk(options['train_folder']))
    for dir_ in dirs:
        files = glob.glob(os.path.join(root, dir_, options['tmp_folder'])+'/*.nii.gz')
        print(files)
        for file in files:
            # resize
            data = nib.load(file)
            data = resample_img(data, target_affine=np.eye(3)*2., interpolation='nearest')
            # save new size
            nib.save(data, file)


def split_folds(train_csv_path, seed=300, k_fold=5):
    df = pd.read_csv(train_csv_path)
    skf = StratifiedKFold(n_splits=k_fold, random_state=seed, shuffle=True)

    for i, (train_index, val_index) in enumerate(
        skf.split(df, df["patient_id"])
        ):
        df.loc[val_index, "fold"] = i
    df.reset_index(inplace=True)
    df.drop(columns=['index'], inplace=True)
    df.to_csv(train_csv_path, index=False)
