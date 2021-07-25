# -*- coding: utf-8 -*-
# ***********************************************
# Configuration for the both training and testing
# ***********************************************
class Configuration():
    '''
    Models and data initialization
    '''
    checkpoint_dir = ''
    data_dir = './Datasets_Makeup'
    model_dir = './models_meta_tar_ir152' # Save dir for Adv-Makeup checkpoints
    after_dir = 'after_aligned_600'
    before_dir = 'before_aligned_600'
    lmk_name = 'landmark_aligned_600.pk' # Landmarks for un- and real-world makeup faces
    use_se = True
    pretrained = False
    train_model_name_list = ['irse50', 'facenet', 'mobile_face']
    val_model_name_list = ['ir152']

    '''
    Params for the model training and testing
    '''
    lr = 0.001 # Learning rate
    update_lr_m = 0.001 # Learning rate for meta-optimization
    epoch_steps = 300
    gpu = 0
    n_threads = 8
    batch_size = 4
    input_dim = 3

    '''
    Input data preprocessing
    '''
    resize_size = (420, 160) # Size of input eye-area
    # Idxes of the eye-area in the facial landmark list
    eye_area = [9, 10, 11, 19, 84, 29, 79, 28, 24, 73, 70, 75, 74, 13, 15, 14, 22]
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
