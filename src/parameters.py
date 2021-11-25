class Parameters:
    training_parameters = {
        'mixed_precision': True,
        'epochs': 250,
        'loss': 'bce',
        'kl_beta': 1.,
    }
    optimizer_parameters = {
        'optimizer': 'Adam',
        'lr': 1e-3,
        'wd': 1e-5,
        'train_bs': 4,
        'val_bs': 32
    }
    model_parameters = {
        'model': 'unet',
        'input_channels': 1,
        'output_channels': 1,
        'decoder_output_channels': 32,
        'semantic_classes': 2,
        'stem_filters': 32,
        'unet_encoder': 'rexnet_200',
        'squeeze_bottleneck': False,
        'num_classes': 100,
        'inception_style_decoder': False,
        'dilated_residues': False,
        'classifier': True,
        'classifier_pooling': 0
    }
    logging_parameters = {
        'log_path': '/ayb/vol1/datasets/chest/to_paper/finetuner/'
    }
    eval_parameters = {
        'threshold_area': 500,
    }
    prior_net_parameters = {
        'hidden_dim': 6,
        'mix_hidden_repr': True
    }
    uncertainty_parameters = {
        'estimate_uncertainty': False,
        'dropout_p': 0.2
    }
    dataset_parameters = {
        'experts': '123'
    }

    @classmethod
    def get_attrs(cls):
        ans_dict = dict()
        relevant_attrs = [elem for elem in dir(cls) if '__' not in elem and elem != 'get_attrs']
        return {elem:eval('Parameters.'+elem) for elem in relevant_attrs}
        
        