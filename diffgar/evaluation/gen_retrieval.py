import torch
import numpy as np
import itertools
from tqdm import tqdm


def generate_from_prompts(model, prompts, **kwargs):

    preds = model.inference(prompts, model.inference_scheduler, **kwargs)

    return preds


def get_embeddings_and_preds(model, datum,preextracted_features=True, **kwargs):

    prompt = datum['prompt']

    prompt = [prompt] if isinstance(prompt, str) else prompt

    audio = datum.get('audio', None)

    text_embeds = model.encoder_pair.get_text_embedding(prompt)
    audio = audio if preextracted_features else model.encoder_pair.get_audio_embedding_from_data(audio)

    audio = torch.stack(audio) if isinstance(audio, list) else audio

    preds = generate_from_prompts(
        model, prompts=prompt, **kwargs).permute(0, 2, 1)

    return {
        'text_embeds': text_embeds,
        'audio_embeds': audio,
        'preds': preds
    }


# clap score computation between:
# - ground truth text and audio embeddings
# - predicted audio embedding and ground truth text embeddings
# - predicted audio embedding and ground truth audio embeddings

def compute_sims(text_embeds, audio_embeds, preds):

    print(f"text_embeds: {text_embeds.shape}")
    print(f"audio_embeds: {audio_embeds.shape}")
    print(f"preds: {preds.shape}")
    
    audio_embeds = audio_embeds.mean(dim=1)
    preds = preds.mean(dim=1)
    
    
    # if the embedding dimensions are the same for text embeds and audio embeds, we can compute the similarities directly


    retrieve_gt_text_from_gt_audio = None
    retrieve_gt_audio_from_gt_text = None
    retrieve_gt_text_from_pred_audio = None

    if text_embeds.shape[-1] == audio_embeds.shape[-1]:

        retrieve_gt_audio_from_gt_text = text_embeds @ audio_embeds.t()
        retrieve_gt_text_from_gt_audio = audio_embeds @ text_embeds.t()
        retrieve_gt_text_from_pred_audio = preds @ text_embeds.t()
        
        
            
        print(f"retrieve_gt_audio_from_gt_text: {retrieve_gt_audio_from_gt_text.shape}")
        print(f"retrieve_gt_text_from_gt_audio: {retrieve_gt_text_from_gt_audio.shape}")
        print(f"retrieve_gt_text_from_pred_audio: {retrieve_gt_text_from_pred_audio.shape}")
    
    retrieve_gt_audio_from_pred_text = preds @ audio_embeds.t()
    #instead of dot-product, we can use cosine similarity
    
    # retrieve_gt_audio_from_pred_text = audio_embeds @ audio_embeds.t()

    
    print(f"retrieve_gt_audio_from_pred_text: {retrieve_gt_audio_from_pred_text.shape}")


    
    out_ = {
        'retrieve_gt_text_from_gt_audio': retrieve_gt_text_from_gt_audio,
        'retrieve_gt_audio_from_gt_text': retrieve_gt_audio_from_gt_text,
        'retrieve_gt_text_from_pred_audio': retrieve_gt_text_from_pred_audio,
        'retrieve_gt_audio_from_pred_text': retrieve_gt_audio_from_pred_text
    }

    for k, v in out_.items():
        print(f"{k}: {v.shape}") if v is not None else None

    return out_


def compute_clap_score(sims_dict):

    # gt_text_audio_sims_avg = sims_dict['gt_text_audio_sims_avg']
    # gt_text_audio_sims_max = sims_dict['gt_text_audio_sims_max']
    # pred_audio_gt_text_sims_avg = sims_dict['pred_audio_gt_text_sims_avg']
    # pred_audio_gt_text_sims_max = sims_dict['pred_audio_gt_text_sims_max']
    # pred_audio_gt_audio_sims_avg = sims_dict['pred_audio_gt_audio_sims_avg']
    # pred_audio_gt_audio_sims_max = sims_dict['pred_audio_gt_audio_sims_max']


    retrieve_gt_text_from_gt_audio = sims_dict['retrieve_gt_text_from_gt_audio']
    retrieve_gt_audio_from_gt_text = sims_dict['retrieve_gt_audio_from_gt_text']
    retrieve_gt_text_from_pred_audio = sims_dict['retrieve_gt_text_from_pred_audio']
    retrieve_gt_audio_from_pred_text = sims_dict['retrieve_gt_audio_from_pred_text']

    # gt_text_audio_sims_avg_diag = torch.diag(gt_text_audio_sims_avg)
    # gt_text_audio_sims_max_diag = torch.diag(gt_text_audio_sims_max)

    # pred_audio_gt_text_sims_avg_diag = torch.diag(pred_audio_gt_text_sims_avg)
    # pred_audio_gt_text_sims_max_diag = torch.diag(pred_audio_gt_text_sims_max)

    # pred_audio_gt_audio_sims_avg_diag = torch.diag(
    #     pred_audio_gt_audio_sims_avg)
    # pred_audio_gt_audio_sims_max_diag = torch.diag(
    #     pred_audio_gt_audio_sims_max)
    
    
    retrieve_gt_text_from_gt_audio_diag = torch.diag(retrieve_gt_text_from_gt_audio) if retrieve_gt_text_from_gt_audio is not None else None
    retrieve_gt_audio_from_gt_text_diag = torch.diag(retrieve_gt_audio_from_gt_text) if retrieve_gt_audio_from_gt_text is not None else None
    retrieve_gt_text_from_pred_audio_diag = torch.diag(retrieve_gt_text_from_pred_audio) if retrieve_gt_text_from_pred_audio is not None else None
    retrieve_gt_audio_from_pred_text_diag = torch.diag(retrieve_gt_audio_from_pred_text) if retrieve_gt_audio_from_pred_text is not None else None
    

    # clap_score = {
    #     'diagonals': {
    #         'gt_text_audio_sims_CLAP_avg': gt_text_audio_sims_avg_diag.mean(),
    #         'gt_text_audio_sims_CLAP_max': gt_text_audio_sims_max_diag.mean(),
    #         'pred_audio_gt_text_sims_CLAP_avg': pred_audio_gt_text_sims_avg_diag.mean(),
    #         'pred_audio_gt_text_sims_CLAP_max': pred_audio_gt_text_sims_max_diag.mean(),
    #         'pred_audio_gt_audio_sims_CLAP_avg': pred_audio_gt_audio_sims_avg_diag.mean(),
    #         'pred_audio_gt_audio_sims_CLAP_max': pred_audio_gt_audio_sims_max_diag.mean()},
    #     'averages': {
    #         'gt_text_audio_sims_CLAP_avg': (gt_text_audio_sims_avg - torch.diag(gt_text_audio_sims_avg_diag)).mean(),
    #         'gt_text_audio_sims_CLAP_max': (gt_text_audio_sims_max - torch.diag(gt_text_audio_sims_max_diag)).mean(),
    #         'pred_audio_gt_text_sims_CLAP_avg': (pred_audio_gt_text_sims_avg - torch.diag(pred_audio_gt_text_sims_avg_diag)).mean(),
    #         'pred_audio_gt_text_sims_CLAP_max': (pred_audio_gt_text_sims_max - torch.diag(pred_audio_gt_text_sims_max_diag)).mean(),
    #         'pred_audio_gt_audio_sims_CLAP_avg': (pred_audio_gt_audio_sims_avg - torch.diag(pred_audio_gt_audio_sims_avg_diag)).mean(),
    #         'pred_audio_gt_audio_sims_CLAP_max': (pred_audio_gt_audio_sims_max - torch.diag(pred_audio_gt_audio_sims_max_diag)).mean()
            
    #     }
    # }
    
    clap_score = {
        'diagonals': {
            'retrieve_gt_text_from_gt_audio_CLAP': retrieve_gt_text_from_gt_audio_diag.mean().item() if retrieve_gt_text_from_gt_audio_diag is not None else None,
            'retrieve_gt_audio_from_gt_text_CLAP': retrieve_gt_audio_from_gt_text_diag.mean().item() if retrieve_gt_audio_from_gt_text_diag is not None else None,
            'retrieve_gt_text_from_pred_audio_CLAP': retrieve_gt_text_from_pred_audio_diag.mean().item() if retrieve_gt_text_from_pred_audio_diag is not None else None,
            'retrieve_gt_audio_from_pred_text_CLAP': retrieve_gt_audio_from_pred_text_diag.mean().item() if retrieve_gt_audio_from_pred_text_diag is not None else None
        },
        'averages': {
            'retrieve_gt_text_from_gt_audio_CLAP': (retrieve_gt_text_from_gt_audio - torch.diag(retrieve_gt_text_from_gt_audio_diag)).mean().item() if retrieve_gt_text_from_gt_audio_diag is not None else None,
            'retrieve_gt_audio_from_gt_text_CLAP': (retrieve_gt_audio_from_gt_text - torch.diag(retrieve_gt_audio_from_gt_text_diag)).mean().item() if retrieve_gt_audio_from_gt_text_diag is not None else None,
            'retrieve_gt_text_from_pred_audio_CLAP': (retrieve_gt_text_from_pred_audio - torch.diag(retrieve_gt_text_from_pred_audio_diag)).mean().item() if retrieve_gt_text_from_pred_audio_diag is not None else None,
            'retrieve_gt_audio_from_pred_text_CLAP': (retrieve_gt_audio_from_pred_text - torch.diag(retrieve_gt_audio_from_pred_text_diag)).mean().item() if retrieve_gt_audio_from_pred_text_diag is not None else None
        }
    }

    return clap_score


def compute_retrieval_metrics(query_key_sim, ground_truth_idx, ks=[1, 3, 5, 10]):

    

    metrics = {f"mean_rank": 0, f"median_rank": 0}
    
    metrics['Recall'] = {}
    metrics['Precision'] = {}
    metrics['mAP'] = {}
    
    for k in ks:
        metrics['Recall'][k] = 0
        metrics['Precision'][k] = 0
        metrics['mAP'][k] = 0
    
    ranks_ = []

    # Iterate over each query
    
    for i in range(query_key_sim.shape[0]):
        ground_truth_idxx = torch.tensor(ground_truth_idx[i]).unsqueeze(-1)
        ranking = torch.argsort(query_key_sim[i], descending=True)
        # print(f"ranking: {ranking}")
        # print(f"ground_truth_idxx: {ground_truth_idxx}")
        
        ranks = torch.isin(ranking, ground_truth_idxx).cpu().numpy()
        ranks = np.where(ranks)[0]

        # Rank Metrics
        ranks_.append(ranks)
        
        # Precision, Recall, and mAP
        for k in ks:
            relevant_in_top_k = np.sum(ranks < k)
            total_relevant = len(ground_truth_idx[i])
            
            metrics['Recall'][k] += relevant_in_top_k / total_relevant  # Recall@k
            metrics['Precision'][k] += relevant_in_top_k / k  # Precision@k

            # mAP@k
            precisions = [(r < k) * (1 / (r + 1)) for r in ranks]
            if len(precisions) > 0:
                metrics['mAP'][k] += np.sum(precisions) / min(k, total_relevant)
            else:
                metrics['mAP'][k] += 0.0

    
    num_queries = query_key_sim.shape[0]

    # get the mean rank and median rank from ranks_
    ranks_ = np.concatenate(ranks_)
    metrics['mean_rank'] = np.mean(ranks_ + 1)
    metrics['median_rank'] = np.floor(np.median(ranks_)) + 1

    for key in metrics.keys():
        if key not in ['mean_rank', 'median_rank']:
            for k in ks:
                metrics[key][k] /= num_queries
                
    print(metrics)
    return metrics


@torch.no_grad()
def eval_dataset(model, dataset, limit_n=-1, preextracted_features=True, strict_retrieval = False, **kwargs):

    model.eval()

    all_metrics = {}

    out_ = pred_dataset(
        model=model,
        dataset=dataset,
        limit_n=limit_n,
        preextracted_features=preextracted_features,
        **kwargs
    )

    audio_embeds, text_embeds, preds, file_idx = out_['audio_embeds'], out_[
        'text_embeds'], out_['preds'], out_['file_idx']

    audio_embeds = torch.stack(audio_embeds).cpu()
    text_embeds = torch.cat(text_embeds).cpu()
    preds = torch.cat(preds).cpu()

    sims_dict = compute_sims(text_embeds, audio_embeds, preds)
    clap_score = compute_clap_score(sims_dict)

    all_metrics.update(clap_score)
    
    file_idx = [[x] for x in list(range(len(file_idx)))] if strict_retrieval else file_idx

    # if we're retrieving from text to audio or audio to text, the gt idx is the file idx

    # for key in ['gt_text_audio_sims_avg', 'gt_text_audio_sims_max', 'pred_audio_gt_text_sims_avg', 'pred_audio_gt_text_sims_max', 'pred_audio_gt_audio_sims_avg', 'pred_audio_gt_audio_sims_max']:
        
    
    to_compute_ = {
        x:sims_dict[x] for x in ['retrieve_gt_audio_from_gt_text',
                    'retrieve_gt_text_from_gt_audio',
                    'retrieve_gt_text_from_pred_audio',
                    'retrieve_gt_audio_from_pred_text'] if sims_dict[x] is not None
        }
        
    for key in to_compute_.keys():
        retrieval_metrics = compute_retrieval_metrics(to_compute_[key], file_idx, ks=[1, 3, 5, 10])

        all_metrics.update({
            key : retrieval_metrics
        })

    return all_metrics


def pred_dataset(model, dataset,limit_n=-1, preextracted_features=True, **kwargs):

    model.eval()

    file_idx = []

    audio_embeds, text_embeds, preds = [], [], []

    captions = []

    for i, datum in tqdm(itertools.islice(enumerate(dataset), 0, limit_n)):

        embeddings_and_preds = get_embeddings_and_preds(
            model, datum, preextracted_features, **kwargs)
        
        audio_embeds.append(embeddings_and_preds['audio_embeds']) if isinstance(
            embeddings_and_preds['audio_embeds'], torch.Tensor) else embeddings_and_preds['audio_embeds']['embedding_proj']
        
        text_embeds.append(
            embeddings_and_preds['text_embeds'].get('projected_pooler_output',torch.zeros(1,1)))
        preds.append(embeddings_and_preds['preds'])
        file_idx.append(datum['file_idx'])
    
    ##file_idx can contain multiple times the same idx. make a new list that contains
    ## the list of idxs that contain the same idx
    
    # for example, [1, 1, 2, 2, 3, 3, 4, 4, 4] -> [[0,1], [0,1], [2,3], [2,3], [4,5], [4,5], [6,7,8], [6,7,8], [6,7,8]]
    
    file_idx = [[i for i, x in enumerate(file_idx) if x == idx] for idx in file_idx]


    return {
        'audio_embeds': audio_embeds,
        'text_embeds': text_embeds,
        'preds': preds,
        'file_idx': file_idx
    }
