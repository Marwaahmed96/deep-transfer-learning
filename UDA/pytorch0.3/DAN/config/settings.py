# --------------------------------------------------------------------------------
# configuration file
#
# Sergi Valverde 2017
# --------------------------------------------------------------------------------
# --------------------------------------------------
# Database options
# --------------------------------------------------

# path to training image folder. In all our experiments, training images were inside
# a folder and image modalities were consistent across training images. In the case of
# leave-one-out experiments, the same folder is used


import os
current_path = os.getcwd()
root_path= '/'.join(current_path.split('/')[:current_path.split('/').index('Code')])
datasets_path=os.path.join(root_path, 'DataSets')

class Settings:
    def __init__(self):

        self.options = dict()
        #------------------------
        # Paths
        #------------------------
        
        self.options['root_dir'] = root_path
        self.options['data_path'] = os.path.join(datasets_path, 'ISBIOrig/training')
        self.options['code_path'] = os.path.join(root_path, 'Code/deep-transfer-learning/UDA/pytorch0.3/DAN')
        self.options['train_folder'] = os.path.join(datasets_path, 'ISBI/train/')
        self.options['test_folder'] = os.path.join(datasets_path, 'ISBI/test/')
        self.options["train_csv_path"] = os.path.join(self.options['train_folder'], "train_data.csv")
        self.options['second_train'] = False
        self.options['train_count'] = '2' if self.options['second_train'] else '1'
        self.options['pre_trained_model'] = '2_model.pth'
        # current experiment name
        self.options['experiment'] = 'resnet_DAN_full_isbi_train4'
        self.options["history_csv_path"] = os.path.join(self.options['train_folder'], self.options['experiment'] + '_' + self.options['train_count'] + '_' + "history_data.csv")
        self.options['h5_path'] = os.path.join(datasets_path, 'ISBI/h5df_files/')

        # ------------------------
        # DataBase
        # ------------------------
        
        # image modalities used (T1, FLAIR, PD, T2, ...)
        self.options['modalities'] = ['FLAIR', 'T1']
        # image modalities nifti file names in the same order of options['modalities']
        self.options['x_names'] = ['flair', 'mprage']
        self.options['preprocess_x_names'] = ['FLAIR_brain.nii.gz', 'T1_brain.nii.gz']

        self.options['tmp_folder'] = 'tmp'

        # lesion annotation nifti file names
        self.options['masks'] = ['lesion']
        self.options['y_names'] = ['mask1.nii']
        self.options['preprocess_y_names'] = ['lesion.nii.gz']
        # preprocessing options
        self.options['denoise'] = True
        self.options['denoise_iter'] = 3
        self.options['skull_stripping'] = False
        self.options['register_modalities'] = False
        self.options['bias_corrected'] = False
        self.options['interpolate'] = False
        self.options['interpolate_size'] = (1, 1, 1)
        self.options['save_tmp'] = True
        self.options['debug'] = True
        
        # --------------------------------------------------
        # Model
        # --------------------------------------------------

        self.options['pretrained'] = None
        # percentage of the training vector that is going to be used to validate the model during training
        self.options['train_split'] = 0.25
        # maximum number of epochs used to train the model
        self.options['max_epochs'] = 200
        # maximum number of epochs without improving validation before stopping training
        self.options['patience'] = 40
        # Number of samples used to test at once.
        self.options['batch_size'] = 128
        # verbosity of CNN messaging: 00 (none), 01 (low), 10 (medium), 11 (high)
        self.options['net_verbose'] = 1
        self.options['mode'] = 'cuda1'
        # Select between pixel-wise or fully-convolutional training models. Although implemented, fully-convolutional
        # models have been not tested with this cascaded model
        self.options['fully_convolutional'] = False
        # 3D patch size. So, far only implemented for 3D CNN models.
        self.options['patch_size'] = (16, 16, 16)
        self.options['initial_learning_rate'] = 1e-3
        self.options['learning_rate_drop'] = 0.1  # factor by which the learning rate will be reduced
        # --------------------------------------------------
        # postprocessing
        # --------------------------------------------------
        
        # post-processing binary threshold.
        self.options['t_bin'] = 0.5
        # post-processing minimum lesion size of output candidates
        self.options['l_min'] = 10
        self.options['min_error'] = 0.5

        # --------------------------------------------------
        # Experiment options
        # --------------------------------------------------

        # minimum threshold used to select candidate voxels for training. Note that images are
        # normalized to 0 mean 1 standard deviation before thresholding. So a value of t > 0.5 on FLAIR is
        # reasonable in most cases to extract all WM lesion candidates
        self.options['min_th'] = 0.5

        # randomize training features before fitting the model.
        self.options['randomize_train'] = True

        self.options['seed'] = 55
        self.options['k_fold'] = 5

        # --------------------------------------------------
        # model parameters
        # --------------------------------------------------
        # model config
        self.options['channels'] = len(self.options['modalities'])
        self.options['out_channels'] = 1
        self.options['input_shape'] = (*self.options['patch_size'], self.options['channels'])
        self.options['depth'] = 4 # depth of layers for V/Unet
        self.options['n_base_filters'] = 32
        self.options['pooling_kernel'] = (2, 2, 2)  # pool size for the max pooling operations
        self.options['deconvolution'] = True  # if False, will use upsampling instead of deconvolution

        # model train config
        # file paths to store the network parameter weights. These can be reused for posterior use.
        self.options['weight_paths'] = os.path.join(self.options['code_path'], 'weights')
        # where the model weights initialization so each time begin with the same weight to compare between different models
        self.options['initial_weights_path'] = os.path.join(self.options['weight_paths'], 'initial_weights.hdf5')
        self.options['load_initial_weights'] = True
        # Where to save the model weights during train
        # ,TPR,FPR,FNR,Tversky,dice_coefficient
        self.options['metrics'] = ['mse']

    # @staticmethod
    def get_options(self):
        return self.options




