## make this scriptable later onfrom diffgar.models.ldm.diffusion import DiffGarLDM

from diffgar.models.ldm.diffusion import DiffGarLDM

from diffgar.evaluation.gen_retrieval import *
from diffgar.dataloading.dataloaders import *

import wandb


def load_model_and_dataset_eval(model_name, task, device='cuda:4'):
    
    path = f's3://maml-aimcdt/storage/julien/DiffGAR/training_checkpoints/{model_name}'
    experiment_name = path.split('/')[-1]
    config_path = path + '/config.yaml'
    ckpt_path = path + '/checkpoint-step=100000-recent.ckpt'


    ## get the config from wandb
    api = wandb.Api()
    runs = api.runs(f'jul-guinot/DiffGAR-LDM')
    ## get the config where names match the experiment name
    for run in runs:
        if run.name == experiment_name:
            config = run.config

    training_encoder_pair = config['model']['encoder_pair']
    
    encoder_pair_to_new_dir = {
        'song_describer': {
        'muleT5':   '/import/research_c4dm/jpmg86/song-describer/data/mule/npy/1hz',
        'clap':    '/import/research_c4dm/jpmg86/song-describer/data/npy'
        }
    }
    
    encoder_pair_to_old_dir = {
        'song_describer': {
        'muleT5':   '/import/research_c4dm/jpmg86/song-describer/data/audio',
        'clap':    '/import/research_c4dm/jpmg86/song-describer/data/audio'
        }
    }



    model = DiffGarLDM.from_pretrained(config_path, ckpt_path, device=device)

    latent_dm = TextAudioDataModule(
        task=task,
        batch_size=2,
        target_n_samples=480000,
        preextracted_features=True,
        target_sr=48000,
        truncate_preextracted=64,
        new_dir=encoder_pair_to_new_dir[task][training_encoder_pair],
        root_dir=encoder_pair_to_old_dir[task][training_encoder_pair]
        )
    latent_dm.setup(None)
    
    if task=='song_describer':
        dataset = latent_dm.val_dataset
    else:
        dataset = latent_dm.test_dataset #might need to modify musiccaps so that it is all in the test set

    return model, dataset, experiment_name, config



def run_eval(guidance_scales, model_names, task, log=True, device='cuda:4'):
    
    for model_name in model_names:
        model, dataset, experiment_name, config = load_model_and_dataset_eval(model_name, task, device=device)

        for guidance_scale in guidance_scales:
            out_ = eval_dataset(model,dataset, limit_n=len(dataset), disable_progress=True, num_steps = 50, strict_retrieval=True, guidance_scale = guidance_scale)


            def log_results(data_dict, experiment_config = config, experiment_name = experiment_name, task = task, guidance_scale = guidance_scale):
                # Extract CLAP scores from 'diagonals' and 'averages' into a DataFrame
                training_guidance = experiment_config['model']['unet_model_config'].get('classifier_free_guidance_strength', 0.0)
                inference_guidance = guidance_scale
                original_name = experiment_name 
                
                training_config = experiment_config
                
                
                project = 'DiffGAR-LDM-retrieval-eval'
                
                config = {
                    'training_guidance': training_guidance,
                    'inference_guidance': inference_guidance,
                    'task': task,
                    'experiment_name': experiment_name,
                    'original_name': original_name,
                    'training_config': training_config
                }
                
                for key in data_dict.keys():
                    try:
                        retrieval_metrics = data_dict[key]
                        ks = [1,3,5,10]
                        for k in ks:
                            new_dict = {}
                            for metric, values in retrieval_metrics.items():
                                if isinstance(values, dict) and values[k] is not None:
                                    new_dict[metric] = round(values[k], 2)
                                else:
                                    new_dict[metric] = round(values, 2) if values is not None else None
                            
                            
                            config.update({
                                'k': k,
                                'retrieval_metric': key
                            })
                            new_name = f'{original_name} - {key} - k={k} - guidance={inference_guidance} - task={task} - training_guidance={training_guidance}'
                            
                            if log:
                                wandb.init(project=project, name=new_name, config=config)
                                wandb.log(new_dict)
                                wandb.finish()
                            
                    except Exception as e:
                        
                        print(f'Failed to log {key}, {e}')
                        print(data_dict[key])
                
            log_results(out_)
        

if __name__ == '__main__':
    # guidance_scales = [0,0.1,0.3,0.5,1,5,10]
    guidance_scales = [3]
    
    experiments = {
        'base' : {
            'task' : {
                'upmm' : {
                    # 'CLAPT5' : {'model_name' : 'diffgar-training-2024-10-12-00-59-43-7lnzqj-ip-10-2-239-154.ec2.internal'},
                    # 'CLAPCLAP' : {'model_name' : 'diffgar-training-2024-10-09-15-32-35-9ngrhp-ip-10-0-73-91.ec2.internal'},
                    # 'MULET5' : '',
                },
                'song_describer' : {
                    # 'CLAPT5' : {'model_name' : 'diffgar-training-2024-10-12-00-32-45-9i65xk-ip-10-0-73-210.ec2.internal'},
                    # 'CLAPCLAP' : {'model_name' : 'diffgar-training-2024-10-08-15-39-16-394tsk-ip-10-2-207-31.ec2.internal'},
                    # 'MULET5' : {'model_name' : 'diffgar-training-2024-10-20-09-21-57-iv6e5u-ip-10-0-109-23.ec2.internal'}
                }
            }
        },
        'model_scale' : {
            'task' : {
                'upmm' : {
                    'CLAPT5' : {
                        # 'xlarge' : '',
                        # 'large' : '',
                        # 'base' : {'model_name' : 'diffgar-training-2024-10-10-08-56-46-3a54dk-ip-10-0-136-252.ec2.internal'},
                        # 'small' : '',
                        # 'tiny' : '',
                    },
                    'CLAPCLAP' : {
                        # 'xlarge' : '',
                        # 'large' : '',
                        # 'base' : {'model_name' : 'diffgar-training-2024-10-09-15-48-35-v5sxgr-ip-10-0-86-191.ec2.internal'},
                        # 'small' : '',
                        # 'tiny' : '',
                    },
                    'MULET5' : {
                        # 'xlarge' : '',
                        # 'large' : '',
                        'base' : {'model_name':'diffgar-training-2024-10-21-13-00-40-7w1jx7-ip-10-2-86-220.ec2.internal'},
                        # 'small' : {'model_name':'diffgar-training-2024-10-20-13-34-10-xnyaqm-ip-10-2-81-46.ec2.internal'},
                        # 'tiny' : {'model_name':'diffgar-training-2024-10-20-12-07-59-ikd551-ip-10-0-110-34.ec2.internal'},
                    },
                },
                'song_describer' : {
                    'CLAPT5' : {
                        # 'xlarge' : '',
                        # 'large' : '',
                        # 'base' : {'model_name' : 'diffgar-training-2024-10-08-16-11-19-p6io6d-ip-10-0-156-223.ec2.internal'},
                        # 'small' : '',
                        # 'tiny' : '',
                    },
                    'CLAPCLAP' : {
                        # 'xlarge' : '',
                        # 'large' : '',
                        # 'base' : {'model_name' : 'diffgar-training-2024-10-09-08-20-15-pwappv-ip-10-0-121-167.ec2.internal'},
                        # 'small' : '',
                        # 'tiny' : '',
                    },
                    'MULET5' : {
                        # 'xlarge' : '',
                        # 'large' : '',
                        # 'base' : {'model_name':'diffgar-training-2024-10-20-09-21-57-iv6e5u-ip-10-0-109-23.ec2.internal'},
                        # 'small' : {'model_name':'diffgar-training-2024-10-20-10-04-37-nf0eca-ip-10-2-89-112.ec2.internal'},
                        # 'tiny' : {'model_name':'diffgar-training-2024-10-20-09-54-08-r9wzby-ip-10-0-199-175.ec2.internal'},
                    },
                },
            },
        },
        # 'guidance' : {
        #     'task' : {
        #         'upmm' : {
        #             'CLAPT5' : {
        #                 '0' : '',
        #                 '0.1' : '',
        #                 '0.3' : '',
        #                 '0.5' : '',
        #                 '1' : '',
        #             },
        #             'CLAPCLAP' : {
        #                 '0' : '',
        #                 '0.1' : '',
        #                 '0.3' : '',
        #                 '0.5' : '',
        #                 '1' : '',
        #             },
        #             'MULET5' : {
        #                 '0' : '',
        #                 '0.1' : '',
        #                 '0.3' : '',
        #                 '0.5' : '',
        #                 '1' : '',
        #             },
        #         },
        #         'song_describer' : {
        #             'CLAPT5' : {
        #                 '0' : '',
        #                 '0.1' : '',
        #                 '0.3' : '',
        #                 '0.5' : '',
        #                 '1' : '',
        #             },
        #             'CLAPCLAP' : {
        #                 '0' : '',
        #                 '0.1' : '',
        #                 '0.3' : '',
        #                 '0.5' : '',
        #                 '1' : '',
        #             },
        #             'MULET5' : {
        #                 '0' : '',
        #                 '0.1' : '',
        #                 '0.3' : '',
        #                 '0.5' : '',
        #                 '1' : '',
        #             },
        #         },
        #     },
        # },
    }
    
    tasks = [
        'song_describer',
        # 'musiccaps'
        ]
    
    #get all the experiments from the dict and build a model_names list
    
    def get_models_to_run(dict_):
        model_names = []
        for key, value in dict_.items():
            if key == 'model_name':
                model_names.append(value)
                
            elif isinstance(value, dict):
                model_names.extend(get_models_to_run(value))
                
        return model_names
    
    model_names = get_models_to_run(experiments)
    
    print(model_names)
    
    for task in tasks:
        run_eval(guidance_scales, model_names, task, log=False, device='cuda:7')