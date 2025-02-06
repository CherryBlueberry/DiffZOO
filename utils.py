import torch
import os, random, math
import numpy as np
import math
from transformers import BertTokenizer, BertForMaskedLM
from dm_utils import query_SLD
import threading


def seed_all(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def adjust_learning_rate(lr, epoch, epochs, lr_step_mailstone=[10], lr_decay_rate=0.1, lr_schedule='step'):
    if lr_schedule == 'cosine':
        eta_min = lr * (lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / epochs)) / 2
        return lr
    elif lr_schedule == 'step':
        if epoch <= lr_step_mailstone[0]:
            return lr
        elif epoch > lr_step_mailstone[0] and lr <= lr_step_mailstone[1]:
            return lr*0.1
    else:
        raise NotImplementedError


def zero_one_norm(x):
    x = torch.clamp(x, min=-1., max=1.)
    x = (x-(-1.))/2.
    return x


def get_sample_z(z):
    z_discrete = np.zeros(z.shape)
    w = z.shape[0]
    z_norm = zero_one_norm(z.clone())
    for i in range(w):
        p = [z_norm[i].item(), 1.-z_norm[i].item()]   
        z_discrete[i] = np.random.choice([1, 0], p = p)

    return z_discrete


def get_sample_u(u):
    u_discrete = np.zeros(u.shape)
    u_norm = zero_one_norm(u.clone())
    h, w = u.shape
    for i in range(w):
        u_vector = u_norm.detach().cpu().numpy()[:, i]
        p_vector = u_vector/np.sum(u_vector)
        u_discrete_vector_idx = np.random.choice(range(h), p = p_vector)
        u_discrete[u_discrete_vector_idx, i] = 1

    return u_discrete


def get_prompt(z_discrete, u_discrete, candidate, token, tokenizer):
    if np.sum(z_discrete) == 0:
        return tokenizer.decode(token)
    else:
        index = np.where(z_discrete==1.)[0]

        for i in index:
            u_vecter = u_discrete[:, i]
            candidate_idx = np.where(u_vecter==1.)[0]
            _, candidate_token, _ = candidate[int(i)][int(candidate_idx)]
            token[i] = candidate_token

    prompt = tokenizer.decode(token)

    return prompt


def check_nan_torch(x):
    if torch.sum(torch.isnan(x))>0:
        return True
    else:
        return False


def check_nan_numpy(x):
    if np.sum(np.isnan(x))>0:
        return True
    else:
        return False


def get_up_and_down(z, u, delta = 1e-5):
    assert len(z) == len(u), 'error'
    z_up = []
    z_down = []
    u_up = []
    u_down = []
    z_delta_list = []
    u_delta_list = []

    for i in range(len(z)):
        z_delta = torch.rand(z[i].shape)*delta + 1e-8
        u_delta = torch.rand(u[i].shape)*delta + 1e-8
        z_up.append(z[i]+z_delta)
        z_down.append(z[i]-z_delta)
        u_up.append(u[i]+u_delta)
        u_down.append(u[i]-u_delta)
        z_delta_list.append(z_delta)
        u_delta_list.append(u_delta)

    return z_up, z_down, u_up, u_down, z_delta_list, u_delta_list


def success(out, unsafe_labels, unsafe_threshold=0.0):
    unsafe_cnt = 0
    for d in out:
        if d['class'] in unsafe_labels:
            if d['score'] >= unsafe_threshold:
                unsafe_cnt += 1
    if unsafe_cnt > 0:
        return True
    else:
        return False


def get_ASR(ASR_list, bs):
    success_num = 0.
    for i in ASR_list:
        success_num += i*bs
    return success_num/(bs*len(ASR_list))


def val_job(j, cuda_num, pip_list, z, u, candidate, unsafe_token, tokenizer, seed, out_list, detector):
    c = j%cuda_num
    device = torch.device(f"cuda:{c}")
    safe_pipe = pip_list[c]

    print(f'Valudation {j}/{cuda_num-1} on device {c} beginning...')
    
    sample_z = get_sample_z(z)
    sample_u = get_sample_u(u)
    sample_prompt = get_prompt(sample_z, sample_u, candidate, unsafe_token, tokenizer)

    sample_out = query_SLD(prompt=sample_prompt, safe_pipe=safe_pipe, sdseed=seed, device=device, sub_defense='strong')
    detector_out = detector.detect(np.array(sample_out[0]))
    out_list.append((sample_prompt, sample_out[0], detector_out))


def validation(cuda_num, pip_list, z, u, candidate, unsafe_token, tokenizer, seed, out_list, detector, T):
    threads = []
    for l in range(int(T / cuda_num)):
        for i in range(cuda_num):
            th = threading.Thread(target=val_job, args=(i, cuda_num, pip_list, z, u, candidate, unsafe_token, tokenizer, seed, out_list, detector))
            threads.append(th)
            th.start()

        for th in threads:
            th.join()
        
        threads.clear()

    if T % cuda_num != 0:
        remainder = T % cuda_num
        for i in range(remainder):
            th = threading.Thread(target=val_job, args=(i, cuda_num, pip_list, z, u, candidate, unsafe_token, tokenizer, seed, out_list, detector))
            threads.append(th)
            th.start()

    return out_list


def synsets_bert(prompt, top_k=20):
    tokenizer = BertTokenizer.from_pretrained('./models/Bert')
    model = BertForMaskedLM.from_pretrained('./models/Bert')
    model.eval()

    input_token = tokenizer.encode(prompt, return_tensors="pt")
    mask_id = tokenizer.mask_token_id
    input_list = input_token.clone().detach().numpy()[0, :]

    top_synonyms_ids_list = []

    for i in range(1, len(input_list)-1):
        mask_token = input_token.clone().detach()
        mask_token[0, i] = mask_id
        token_logits = model(mask_token).logits
        mask_token_logits = token_logits[0, i, :]
        top_k_similarity, top_k_tokens = torch.topk(mask_token_logits, top_k, dim=-1)
        top_k_similarity = top_k_similarity.clone().detach().numpy()
        top_k_tokens = top_k_tokens.clone().detach().numpy()
        top_k_words = [tokenizer.decode([token]) for token in top_k_tokens]
        
        ll = []
        for w, id, s in zip(top_k_words, top_k_tokens, top_k_similarity):
            ll.append((w, id, s))

        top_synonyms_ids_list.append(ll)

    return top_synonyms_ids_list


def prepare_log(args):
    if args.attack == 'diffzoo':
        save_dir_name = f'results/{args.defense}_{args.attack}/{args.lr}/{args.T}_{args.K}_{args.P}_{args.S}'
        print(f'===================== Save in                   {save_dir_name}')
        print(f'===================== Defense                   {args.defense}')
        print(f'===================== Attack                    {args.attack}')
        print(f'===================== LR                        {args.lr}')
        print(f'===================== Optimization Step S       {args.S}')
        print(f'===================== Sample prompt Number T    {args.T}')
        print(f'===================== Sample text Number K     {args.K}')
        print(f'===================== ZOO Sample Number P       {args.P}')
        print(f'===================== Seed                      {args.seed}')
    elif args.attack == 'diffzoo-lite':
        save_dir_name = f'results/{args.defense}_{args.attack}/{args.T}_{args.S}'
        print(f'===================== Save in                   {save_dir_name}')
        print(f'===================== Defense                   {args.defense}')
        print(f'===================== Optimization Step S       {args.S}')
        print(f'===================== Sample prompt Number T    {args.T}')
        print(f'===================== Seed                      {args.seed}')
    else:
        raise ValueError
    
    if not os.path.exists(f'./{save_dir_name}'):
        os.makedirs(f'./{save_dir_name}')

    return save_dir_name