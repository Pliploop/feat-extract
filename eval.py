# make this scriptable later onfrom diffgar.models.ldm.diffusion import DiffGarLDM

from diffgar.models.ldm.diffusion import DiffGarLDM
import json


from diffgar.evaluation.gen_retrieval import *
from diffgar.dataloading.dataloaders import *

import pickle
import wandb


def load_model_and_dataset_eval(model_name, model_step, task, device='cuda:4'):

    model_step = model_step if model_step is not None else 100000

    path = f's3://maml-aimcdt/storage/julien/DiffGAR/training_checkpoints/{model_name}'
    experiment_name = path.split('/')[-1]
    print(experiment_name)
    config_path = path + '/config.yaml'
    ckpt_path = path + f'/checkpoint-step={model_step}-recent.ckpt'
    
    print(ckpt_path)

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
            'clap':    '/import/research_c4dm/jpmg86/song-describer/data/clap/npy',
            'MusCALL': '/import/research_c4dm/jpmg86/song-describer/data/muscall/npy/1hz'
        },
        'musiccaps': {
            'muleT5':   '/import/research_c4dm/jpmg86/musiccaps/mule_npy/1hz',
            'clap':    '/import/research_c4dm/jpmg86/musiccaps/clap_npy/1hz',
            'MusCALL': '/import/research_c4dm/jpmg86/musiccaps/muscall/1hz'
        }
    }

    encoder_pair_to_old_dir = {
        'song_describer': {
            'muleT5':   '/import/research_c4dm/jpmg86/song-describer/data/audio',
            'clap':    '/import/research_c4dm/jpmg86/song-describer/data/audio',
            'MusCALL': '/import/research_c4dm/jpmg86/song-describer/data/audio'
        },
        'musiccaps': {
            'muleT5':   '/import/c4dm-datasets/musiccaps/musiccaps_10s',
            'clap':    '/import/c4dm-datasets/musiccaps/musiccaps_10s',
            'MusCALL': '/import/c4dm-datasets/musiccaps/musiccaps_10s'
        }
    }
    
    task_to_task_kws = {
        'musiccaps': {
                'data_path': '/import/c4dm-datasets/musiccaps/musiccaps_10s',
                'csv_path': '/import/c4dm-datasets/musiccaps/musiccaps-public.csv'
        },
        'song_describer': {
                'data_path': '/import/research_c4dm/jpmg86/song-describer/data/audio',
                'csv_path': '/import/research_c4dm/jpmg86/song-describer/data/song_describer.csv'
        }
    }

    print(ckpt_path)
    model = DiffGarLDM.from_pretrained(config_path, ckpt_path, device=device)

    latent_dm = TextAudioDataModule(
        task=task,
        task_kwargs=task_to_task_kws[task],
        batch_size=1,
        preextracted_features=True,
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


def log_results(data_dict, experiment_config=None, experiment_name=None, task=None, guidance_scale=None, log=False, num_samples_per_prompt=None, ks=[1, 3, 5, 10], training_steps=None):
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
        'training_steps': training_steps,
        'experiment_name': experiment_name,
        'num_samples_per_prompt': num_samples_per_prompt,
        'original_name': original_name,
        'training_config': training_config,
        'training_dataset': training_config['data']['task'],
        'model_scale': training_config['model']['unet_model_config']['name'],
        'training_encoder_pair': training_config['model']['encoder_pair'],
        'text_encoder': training_config['model'].get('text_encoder', None),
        'contrastive_loss_weight': training_config['model'].get('contrastive_loss_kwargs', {}).get('weight', None),
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


def run_eval(guidance_scales, model_names,  model_steps, task, log=True, device='cuda:4', distance='cosine', num_samples_per_prompt=[1], limit_n=None, agg = None):

    metrics = []
    sims = []
    out = []
    captions = []
    
    if model_steps is None:
        model_steps = [None] * len(model_names)

    for model_name, model_step in zip(model_names, model_steps):
        model, dataset, experiment_name, config = load_model_and_dataset_eval(
            model_name, model_step, task, device=device)

        limit_n = limit_n if limit_n is not None else len(dataset)

        for guidance_scale in guidance_scales:
            for num_samples_per_prompt in num_samples_per_prompt:
                try:
                    metrics_, sims_, out_, captions_ = eval_dataset(model, dataset, limit_n=limit_n, disable_progress=True, num_steps=50, strict_retrieval=True,
                                                   guidance_scale=guidance_scale, distance=distance, num_samples_per_prompt=num_samples_per_prompt, agg=agg)
                    metrics_ = log_results(metrics_, experiment_config=config, experiment_name=experiment_name,
                                               task=task, guidance_scale=guidance_scale, log=log, num_samples_per_prompt=num_samples_per_prompt,training_steps=model_step)
                    
                    print(json.dumps(metrics_, indent=4))
                    metrics.append(metrics_)
                    config_copy = metrics_.copy()
                    config_copy.pop('metrics')
                    sims.append({**config_copy, 'sims': sims_})
                    out.append({**config_copy, 'out': out_})
                    captions.append({**config_copy, 'captions': captions_})

                except Exception as e:
                    print(
                        f'Failed to evaluate {model_name} with guidance_scale {guidance_scale} and task {task}')
                    raise e

    return metrics, sims, out, captions



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
    parser.add_argument('--save-metrics', type=bool, default=False)
    parser.add_argument('--save-sims', type=bool, default=False)
    parser.add_argument('--save-embeddings', type=bool, default=False)
    parser.add_argument('--save-captions', type=bool, default=False)
    parser.add_argument('--save', type=bool, default=False)
    parser.add_argument('--log', type=bool, default=False)
    parser.add_argument('--file-postfix', type=str, default='')
    args = parser.parse_args()
    device = args.device
    log = args.log
    file_postfix = args.file_postfix
    save_metrics, save_sims, save_embeddings, save_captions, save = args.save_metrics, args.save_sims, args.save_embeddings, args.save_captions, args.save
    if save:
        save_metrics, save_sims, save_embeddings, save_captions = True, True, True, True
    
    print(f'Saving metrics: {save_metrics}')
    print(f'Saving sims: {save_sims}')
    print(f'Saving embeddings: {save_embeddings}')
    
    
    print(f'Running on device {device}')
    

    experiments = {
        'base': {
            'task': {
                'upmm': {
                    'CLAPT5': {'model_name': 'diffgar-training-2024-10-12-00-59-43-7lnzqj-ip-10-2-239-154.ec2.internal'},
                    'CLAPCLAP': {'model_name': 'diffgar-training-2024-10-09-15-32-35-9ngrhp-ip-10-0-73-91.ec2.internal'},
                    'MULET5': {'model_name': 'diffgar-training-2024-10-22-23-39-20-2xguwt-ip-10-2-125-92.ec2.internal'},
                    'MUSCALL': {'model_name': ''},
                    'MUSCALLT5': {'model_name': ''},
                },
                'song_describer': {
                    'CLAPT5': {'model_name': 'diffgar-training-2024-10-12-00-32-45-9i65xk-ip-10-0-73-210.ec2.internal'},
                    'CLAPCLAP': {'model_name': 'diffgar-training-2024-10-08-15-39-16-394tsk-ip-10-2-207-31.ec2.internal'},
                    'MULET5': {'model_name': 'diffgar-training-2024-10-23-10-24-58-ecrjda-ip-10-2-200-242.ec2.internal'},
                    'MUSCALL': {'model_name': ''},
                    'MUSCALLT5': {'model_name': ''},
                }
                
            }
        },
        # 'model_scale': {
        #     'task': {
        #         'upmm': {
        #             'CLAPT5': {
        #                 'xlarge' : {'model_name' : 'diffgar-training-2024-11-19-02-41-34-a877a0-ip-10-2-125-78.ec2.internal'},
        #                 'large' : {'model_name' : 'diffgar-training-2024-11-19-01-53-40-cc99d2-ip-10-2-103-60.ec2.internal'},
        #                 'small' : {'model_name' : 'diffgar-training-2024-11-19-02-57-35-yki4n1-ip-10-0-150-137.ec2.internal'},
        #                 'tiny' : {'model_name' : 'diffgar-training-2024-11-19-02-20-37-nn22pe-ip-10-0-157-183.ec2.internal'},
        #             },
        #             'CLAPCLAP': {
        #                 'xlarge': {'model_name': 'diffgar-training-2024-10-10-08-56-46-3a54dk-ip-10-0-136-252.ec2.internal'},
        #                 'large': {'model_name': 'diffgar-training-2024-10-09-15-48-35-v5sxgr-ip-10-0-86-191.ec2.internal'},
        #                 'small': {'model_name': 'diffgar-training-2024-10-09-15-17-25-8sarg2-ip-10-0-206-118.ec2.internal'},
        #                 'tiny': {'model_name': 'diffgar-training-2024-10-09-16-00-15-1y2807-ip-10-2-224-244.ec2.internal'},

        #             },
        #             'MULET5': {
        #                 'xlarge': {'model_name': 'diffgar-training-2024-10-23-01-08-58-gr6g8k-ip-10-0-125-16.ec2.internal'},
        #                 'large': {'model_name': 'diffgar-training-2024-10-23-00-57-51-azhmyc-ip-10-0-180-242.ec2.internal'},
        #                 'small': {'model_name': 'diffgar-training-2024-10-23-00-41-42-vra4x4-ip-10-2-91-162.ec2.internal'},
        #                 'tiny': {'model_name': 'diffgar-training-2024-10-23-00-07-24-e4fja8-ip-10-2-221-193.ec2.internal'},
        #             },
        #             'MUSCALL': {
        #                 'xlarge': {''},
        #                 'large': {''},
        #                 'small': {''},
        #                 'tiny': {''},
        #             },
        #             'MUSCALLT5': {
        #                 'xlarge': {'model_name': ''},
        #                 'large': {'model_name': ''},
        #                 'small': {''},
        #                 'tiny': {''},
        #             },
        #         },
        #         'song_describer': {
        #             'CLAPT5': {
        #                 # 'xlarge' : {'model_name' : ''},
        #                 # 'large' : {'model_name' : ''},
        #                 'small' : {'model_name' : 'smooth-lake-225'},
        #                 'tiny' : {'model_name' : 'breezy-wind-226'},
        #             },
        #             'CLAPCLAP': {
        #                 'xlarge': {'model_name': 'diffgar-training-2024-10-08-16-11-19-p6io6d-ip-10-0-156-223.ec2.internal'},
        #                 'large': {'model_name': 'diffgar-training-2024-10-09-08-20-15-pwappv-ip-10-0-121-167.ec2.internal'},
        #                 'small': {'model_name': 'diffgar-training-2024-10-08-14-53-57-zqszhl-ip-10-0-154-119.ec2.internal'},
        #                 'tiny': {'model_name': 'diffgar-training-2024-10-08-14-46-11-jwnby2-ip-10-2-94-233.ec2.internal'}
        #             },
        #             'MULET5': {
        #                 # 'xlarge' : {'model_name' : ''},
        #                 # 'large' : {'model_name' : ''},
        #                 'small' : {'model_name' : 'trim-sunset-227'},
        #                 'tiny' : {'model_name' : 'rare-blaze-228'},
        #             },
        #             'MUSCALL': {
        #                 # 'xlarge' : {'model_name' : ''},
        #                 # 'large' : {'model_name' : ''},
        #                 'small' : {'model_name' : ''},
        #                 'tiny' : {'model_name' : ''},
        #             },
        #             'MUSCALLT5': {
        #                 # 'xlarge' : {'model_name' : ''},
        #                 # 'large' : {'model_name' : ''},
        #                 'small' : {'model_name' : ''},
        #                 'tiny' : {'model_name' : ''},
        #             },
        #         },
        #     },
        # },
    }

    tasks = [
        # 'song_describer',
        'musiccaps'
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
    out = {}
    captions = {}
    
    # model_names = [
    #     # 'upbeat-elevator-230'
    # ]

    model_steps = [
        [
            5000,
            10000,
            # 15000,
            # 20000,
            50000,
            # 100000
            ] for _ in model_names
    ] 

    # guidance_scales = [0,0.1,0.3,0.5,1,5,10]
    guidance_scales = [
        3
    ]
    num_samples_per_prompt = [
        1,
        5,
        10,
        # 20,
        # 100
    ]
    for task in tasks:
        for i, model_name in enumerate(model_names):
                # try:  
                for model_steps_ in model_steps[i]:
                    ## check if the experiment is in the json file
                    if os.path.exists(f'results/{task}/metrics{file_postfix}.json'):
                        with open(f'results/{task}/metrics{file_postfix}.json', 'r') as f:
                            data = json.load(f)
                    else:
                        data = {}
                                                
                    new_name = model_name+'-'+str(model_steps_)+'-steps'
                        
                    # if the task is not present or the model is not present
                    if new_name not in data.get(task, {}).keys():
                    # if True:
                        try:
                            metrics_, sims_, out_, captions_ = run_eval(
                                num_samples_per_prompt=num_samples_per_prompt,
                                guidance_scales=guidance_scales,
                                model_names=[model_name],
                                model_steps = [model_steps_],
                                task=task,
                                log=log,
                                device=device,
                                distance='cosine',
                                limit_n=None,
                                agg = None
                            )
                            model_name_ = model_name+'-'+str(model_steps_)+'-steps'
                            
                            try:
                            
                                if task not in metrics:
                                    metrics[task] = {
                                        model_name_: metrics_
                                    }
                                else:
                                    metrics[task].update({
                                        model_name_: metrics_
                                    })
                                
                                if task not in sims:
                                    sims[task] = {
                                        model_name_: sims_
                                    }
                                else:
                                    sims[task].update({
                                        model_name_: sims_
                                    })


                                if task not in out:
                                    out[task] = {
                                        model_name_: out_
                                    }
                                else:
                                    out[task].update({
                                        model_name_: out_
                                    })
                                    
                                    
                                if task not in captions:
                                    captions[task] = {
                                        model_name_: captions_
                                    }
                                    
                                else:
                                    captions[task].update({
                                        model_name_: captions_
                                    })
                            except Exception as e:
                                print(f'Failed to update {model_name} with task {task}')
                                raise e
                            
                            print(json.dumps(captions, indent=4))

                            json_path = f'results/{task}/metrics{file_postfix}.json'
                            pickle_path = f'/import/research_c4dm/jpmg86/DiffGAR/results/{task}/sims{file_postfix}.pkl'
                            embedding_path = f'/import/research_c4dm/jpmg86/DiffGAR/results/{task}/embeddings{file_postfix}.pkl'
                            caption_path = f'results/{task}/captions/captions/{file_postfix}.json'
    

                            update_json_file(json_path, task, model_name_, metrics_) if save_metrics else None
                            update_pickle_file(pickle_path, task, model_name_, sims_) if save_sims else None
                            update_pickle_file(embedding_path, task, model_name_, out_) if save_embeddings else None
                            update_json_file(caption_path, task, model_name_, captions_) if save_captions else None
                        except Exception as e:
                            print(f'Failed to evaluate {model_name} with task {task}')
                            raise e
                    else:
                        print(f'{model_name} already evaluated for task {task}')