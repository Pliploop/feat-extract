# make this scriptable later onfrom diffgar.models.ldm.diffusion import DiffGarLDM

from diffgar.models.ldm.diffusion import DiffGarLDM
import json


from diffgar.evaluation.gen_retrieval import *
from diffgar.dataloading.dataloaders import *

import pickle
import wandb


def load_model_and_dataset_eval(model_name, task, device='cuda:4'):

    path = f's3://maml-aimcdt/storage/julien/DiffGAR/training_checkpoints/{model_name}'
    experiment_name = path.split('/')[-1]
    config_path = path + '/config.yaml'
    ckpt_path = path + '/checkpoint-step=100000-recent.ckpt'

    # get the config from wandb
    api = wandb.Api()
    runs = api.runs(f'jul-guinot/DiffGAR-LDM')
    # get the config where names match the experiment name
    for run in runs:
        if run.name == experiment_name:
            config = run.config

    training_encoder_pair = config['model']['encoder_pair']

    encoder_pair_to_new_dir = {
        'song_describer': {
            'muleT5':   '/import/research_c4dm/jpmg86/song-describer/data/muleproj/npy/1hz',
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

    if task == 'song_describer':
        dataset = latent_dm.val_dataset
    else:
        # might need to modify musiccaps so that it is all in the test set
        dataset = latent_dm.test_dataset

    return model, dataset, experiment_name, config


def log_results(data_dict, experiment_config=None, experiment_name=None, task=None, guidance_scale=None, log=False, num_samples_per_prompt=None, ks=[1, 3, 5, 10]):
    # Extract CLAP scores from 'diagonals' and 'averages' into a DataFrame

    training_guidance = experiment_config['model']['unet_model_config'].get(
        'classifier_free_guidance_strength', 0.0)

    inference_guidance = guidance_scale
    original_name = experiment_name
    training_config = experiment_config

    project = 'DiffGAR-LDM-retrieval-eval'

    config = {
        'training_guidance': training_guidance,
        'inference_guidance': inference_guidance,
        'task': task,
        'experiment_name': experiment_name,
        'num_samples_per_prompt': num_samples_per_prompt,
        'original_name': original_name,
        'training_config': training_config,
        'training_dataset': training_config['data']['task'],
        'model_scale': training_config['model']['unet_model_config']['name'],
        'training_encoder_pair': training_config['model']['encoder_pair'],
        'text_encoder': training_config['model'].get('text_encoder', None),
    }

    config_copy = config.copy()
    config_copy.pop('experiment_name')
    config_copy.pop('training_config')
    config_copy.pop('original_name')


    metrics = {}
    for key in data_dict.keys():
        try:
            metrics[key] = {}
            retrieval_metrics = data_dict[key]
            for k in ks:
                new_dict = {}
                for metric, values in retrieval_metrics.items():
                    if isinstance(values, dict) and values[k] is not None:
                        new_dict[metric] = round(values[k], 2)
                    else:
                        new_dict[metric] = round(
                            values, 2) if values is not None else None

                config.update({
                    'k': k,
                    'retrieval_metric': key
                })
                new_name = f'{original_name} - {key} - k={k} - guidance={inference_guidance} - task={task} - training_guidance={training_guidance}'

                if log:
                    wandb.init(project=project, name=new_name, config=config)
                    wandb.log(new_dict)
                    wandb.finish()

                metrics[key][k] = new_dict
                

        except Exception as e:
            print(f'Failed to log {key}, {e}')
            print(data_dict[key])
            
    config_copy['metrics'] = metrics
    config_copy['training_config'] = None
    return config_copy


def run_eval(guidance_scales, model_names, task, log=True, device='cuda:4', distance='cosine', num_samples_per_prompt=[1], limit_n=None):

    metrics = []
    sims = []

    for model_name in model_names:
        model, dataset, experiment_name, config = load_model_and_dataset_eval(
            model_name, task, device=device)

        limit_n = limit_n if limit_n is not None else len(dataset)

        for guidance_scale in guidance_scales:
            for num_samples_per_prompt in num_samples_per_prompt:
                try:
                    metrics_, sims_ = eval_dataset(model, dataset, limit_n=limit_n, disable_progress=True, num_steps=50, strict_retrieval=True,
                                                   guidance_scale=guidance_scale, distance=distance, num_samples_per_prompt=num_samples_per_prompt)
                    metrics_ = log_results(metrics_, experiment_config=config, experiment_name=experiment_name,
                                               task=task, guidance_scale=guidance_scale, log=log, num_samples_per_prompt=num_samples_per_prompt)
                    
                    
                    metrics.append(metrics_)
                    config_copy = metrics_.copy()
                    config_copy.pop('metrics')
                    sims.append({**config_copy, 'sims': sims_})

                except Exception as e:
                    print(
                        f'Failed to evaluate {model_name} with guidance_scale {guidance_scale} and task {task}')
                    print(e)

    return metrics, sims



def update_json_file(file_path, task, model_name, metrics_):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
    else:
        data = {}

    if task not in data.keys():
        data[task] = {model_name: metrics_}
    else:
        data[task].update({model_name: metrics_})
        
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def update_pickle_file(file_path, task, model_name, sims_):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    else:
        data = {}

    if task not in data.keys():
        data[task] = {model_name: sims_}
    else:
        data[task].update({model_name: sims_})

    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    
    #get the device from the command line --device argument
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:7')
    args = parser.parse_args()
    device = args.device
    
    print(f'Running on device {device}')
    

    experiments = {
        'base': {
            'task': {
                'upmm': {
                    # 'CLAPT5': {'model_name': 'diffgar-training-2024-10-12-00-59-43-7lnzqj-ip-10-2-239-154.ec2.internal'},
                    # 'CLAPCLAP': {'model_name': 'diffgar-training-2024-10-09-15-32-35-9ngrhp-ip-10-0-73-91.ec2.internal'},
                    # 'MULET5': {'model_name': 'diffgar-training-2024-10-22-23-39-20-2xguwt-ip-10-2-125-92.ec2.internal'},
                },
                'song_describer': {
                    # 'CLAPT5': {'model_name': 'diffgar-training-2024-10-12-00-32-45-9i65xk-ip-10-0-73-210.ec2.internal'},
                    'CLAPCLAP': {'model_name': 'diffgar-training-2024-10-08-15-39-16-394tsk-ip-10-2-207-31.ec2.internal'},
                    'MULET5': {'model_name': 'diffgar-training-2024-10-23-10-24-58-ecrjda-ip-10-2-200-242.ec2.internal'},
                }
            }
        },
        'model_scale': {
            'task': {
                'upmm': {
                    'CLAPT5': {
                        # 'xlarge' : {'model_name' : ''},
                        # 'large' : {'model_name' : ''},
                        # 'small' : {'model_name' : ''},
                        # 'tiny' : {'model_name' : ''},
                    },
                    'CLAPCLAP': {
                        # 'xlarge': {'model_name': 'diffgar-training-2024-10-10-08-56-46-3a54dk-ip-10-0-136-252.ec2.internal'},
                        # 'large': {'model_name': 'diffgar-training-2024-10-09-15-48-35-v5sxgr-ip-10-0-86-191.ec2.internal'},
                        # 'small': {'model_name': 'diffgar-training-2024-10-09-15-17-25-8sarg2-ip-10-0-206-118.ec2.internal'},
                        # 'tiny': {'model_name': 'diffgar-training-2024-10-09-16-00-15-1y2807-ip-10-2-224-244.ec2.internal'},

                    },
                    'MULET5': {
                        # 'xlarge': {'model_name': 'diffgar-training-2024-10-23-01-08-58-gr6g8k-ip-10-0-125-16.ec2.internal'},
                        # 'large': {'model_name': 'diffgar-training-2024-10-23-00-57-51-azhmyc-ip-10-0-180-242.ec2.internal'},
                        # 'small': {'model_name': 'diffgar-training-2024-10-23-00-41-42-vra4x4-ip-10-2-91-162.ec2.internal'},
                        # 'tiny': {'model_name': 'diffgar-training-2024-10-23-00-07-24-e4fja8-ip-10-2-221-193.ec2.internal'},
                    },
                },
                'song_describer': {
                    'CLAPT5': {
                        # 'xlarge' : {'model_name' : ''},
                        # 'large' : {'model_name' : ''},
                        # 'small' : {'model_name' : ''},
                        # 'tiny' : {'model_name' : ''},
                    },
                    'CLAPCLAP': {
                        # 'xlarge': {'model_name': 'diffgar-training-2024-10-08-16-11-19-p6io6d-ip-10-0-156-223.ec2.internal'},
                        # 'large': {'model_name': 'diffgar-training-2024-10-09-08-20-15-pwappv-ip-10-0-121-167.ec2.internal'},
                        # 'small': {'model_name': 'diffgar-training-2024-10-08-14-53-57-zqszhl-ip-10-0-154-119.ec2.internal'},
                        # 'tiny': {'model_name': 'diffgar-training-2024-10-08-14-46-11-jwnby2-ip-10-2-94-233.ec2.internal'}
                    },
                    'MULET5': {
                        # 'xlarge' : {'model_name' : ''},
                        # 'large' : {'model_name' : ''},
                        # 'small' : {'model_name' : ''},
                        # 'tiny' : {'model_name' : ''},
                    },
                },
            },
        },
        # 'guidance' : {
        #     'task' : {
        #         'upmm' : {
        #             'CLAPT5' : {
        #                 '0' : '',
        #                 '0.3' : '',
        #                 '0.5' : '',
        #                 '1' : '',
        #             },
        #             'CLAPCLAP' : {
        #                 '0' : '',
        #                 '0.3' : '',
        #                 '0.5' : '',
        #                 '1' : '',
        #             },
        #             'MULET5' : {
        #                 '0' : '',
        #                 '0.3' : '',
        #                 '0.5' : '',
        #                 '1' : '',
        #             },
        #         },
        #         'song_describer' : {
        #             'CLAPT5' : {
        #                 '0' : '',
        #                 '0.3' : '',
        #                 '0.5' : '',
        #                 '1' : '',
        #             },
        #             'CLAPCLAP' : {
        #                 '0' : '',
        #                 '0.3' : '',
        #                 '0.5' : '',
        #                 '1' : '',
        #             },
        #             'MULET5' : {
        #                 '0' : '',
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

    # get all the experiments from the dict and build a model_names list

    def get_models_to_run(dict_):
        model_names = []
        for key, value in dict_.items():
            if key == 'model_name':
                model_names.append(value)

            elif isinstance(value, dict):
                model_names.extend(get_models_to_run(value))

        return model_names

    model_names = get_models_to_run(experiments)

    metrics = {}
    sims = {}
    
    model_names = ['curious-elevator-143']

    # guidance_scales = [0,0.1,0.3,0.5,1,5,10]
    guidance_scales = [
        3
    ]
    num_samples_per_prompt = [
        1,
        5,
        20,
        100
    ]

    for model_name in model_names:
        for task in tasks:
            try:
                metrics_, sims_ = run_eval(
                    num_samples_per_prompt=num_samples_per_prompt,
                    guidance_scales=guidance_scales,
                    model_names=[model_name],
                    task=task,
                    log=False,
                    device=device,
                    distance='cosine',
                    limit_n=None
                )
                if task not in metrics:
                    metrics[task] = {
                        model_name: metrics_
                    }
                else:
                    metrics[task].update({
                        model_name: metrics_
                    })
                
                if task not in sims:
                    sims[task] = {
                        model_name: sims_
                    }
                else:
                    sims[task].update({
                        model_name: sims_
                    })

                update_json_file('results/metrics_contrastive.json', task, model_name, metrics_)
                update_pickle_file('results/sims_contrastive.pkl', task, model_name, sims_)

            except Exception as e:
                print(
                    f'Failed to evaluate {model_name} with task {task} and num_samples_per_prompt {num_samples_per_prompt}')

 

    
    # save sims as pkl
    
        
