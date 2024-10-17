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



    model = DiffGarLDM.from_pretrained(config_path, ckpt_path, device=device)

    latent_dm = TextAudioDataModule(task=task, batch_size=2, target_n_samples=480000, preextracted_features=True, target_sr=48000, truncate_preextracted=64)
    latent_dm.setup(None)
    dataset = latent_dm.val_dataset

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
                                if isinstance(values, dict):
                                    new_dict[metric] = round(values[k], 2)
                                else:
                                    new_dict[metric] = round(values, 2)
                            
                            
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
    guidance_scales = [0,0.1,0.3,0.5,1,5,10]
    # guidance_scales = [3]
    model_names = [
        # base experiments
        'diffgar-training-2024-10-12-00-59-43-7lnzqj-ip-10-2-239-154.ec2.internal', #upmm, guidance 0.1, base, T5, clap
        'diffgar-training-2024-10-12-00-32-45-9i65xk-ip-10-0-73-210.ec2.internal', #songdescriber, guidance 0.1, base, T5, clap
        'diffgar-training-2024-10-09-15-32-35-9ngrhp-ip-10-0-73-91.ec2.internal', #upmm, guidance 0.1, base, clap, clap
        'diffgar-training-2024-10-08-15-39-16-394tsk-ip-10-2-207-31.ec2.internal', #songdescriber, guidance 0.1, base, clap, clap
        # model scale experiments
        # upmm
        # 'diffgar-training-2024-10-10-08-56-46-3a54dk-ip-10-0-136-252.ec2.internal', #upmm, guidance 0.1, xlarge, clap, clap
        # 'diffgar-training-2024-10-09-15-48-35-v5sxgr-ip-10-0-86-191.ec2.internal', #upmm, guidance 0.1, large, clap, clap
        # 'diffgar-training-2024-10-09-15-17-25-8sarg2-ip-10-0-206-118.ec2.internal', #upmm, guidance 0.1, small, clap, clap
        # 'diffgar-training-2024-10-09-16-00-15-1y2807-ip-10-2-224-244.ec2.internal', #upmm, guidance 0.1, tiny, clap, clap
        # '', #upmm, guidance 0.1, xlarge, T5, clap
        # '', #upmm, guidance 0.1, large, T5, clap
        # '', #upmm, guidance 0.1, base, T5, clap
        # '', #upmm, guidance 0.1, small, T5, clap
        # '', #upmm, guidance 0.1, tiny, T5, clap
        # song describer
        # 'diffgar-training-2024-10-08-16-11-19-p6io6d-ip-10-0-156-223.ec2.internal', #songdescriber, guidance 0.1, xlarge, clap, clap
        # 'diffgar-training-2024-10-09-08-20-15-pwappv-ip-10-0-121-167.ec2.internal', #songdescriber, guidance 0.1, large, clap, clap
        # 'diffgar-training-2024-10-08-14-53-57-zqszhl-ip-10-0-154-119.ec2.internal', #songdescriber, guidance 0.1, small, clap, clap
        # 'diffgar-training-2024-10-08-14-46-11-jwnby2-ip-10-2-94-233.ec2.internal', #songdescriber, guidance 0.1, tiny, clap, clap
        # '', #songdescriber, guidance 0.1, xlarge, T5, clap
        # '', #songdescriber, guidance 0.1, large, T5, clap
        # '', #songdescriber, guidance 0.1, base, T5, clap
        # '', #songdescriber, guidance 0.1, small, T5, clap
        # '', #songdescriber, guidance 0.1, tiny, T5, clap
        
        # # guidance experiments, songdescriber or upmm, various guidance scales in [0,0.1,0.3,0.5,1], base, clap, clap
        # '', #upmm, guidance 0.1, base, clap, clap
        # '', #upmm, guidance 0.3, base, clap, clap
        # '', #upmm, guidance 0.5, base, clap, clap
        # '', #upmm, guidance 1, base, clap, clap
        # '', #songdescriber, guidance 0.1, base, clap, clap
        # '', #songdescriber, guidance 0.3, base, clap, clap
        # '', #songdescriber, guidance 0.5, base, clap, clap
        # '', #songdescriber, guidance 1, base, clap, clap
        
        # number of steps but this is for later
        
        
    ]
    task = 'song_describer'
    
    run_eval(guidance_scales, model_names, task, log=True, device='cuda:0')