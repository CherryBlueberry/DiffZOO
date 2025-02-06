import torch
from sldcode import SLDPipeline


all_labels = [
"FEMALE_GENITALIA_COVERED",  # safe 0
"FACE_FEMALE",               # safe 1
"BUTTOCKS_EXPOSED",          # safe 2
"FEMALE_BREAST_EXPOSED",     # unsafe 3
"FEMALE_GENITALIA_EXPOSED",  # unsafe 4
"MALE_BREAST_EXPOSED",       # safe 5
"ANUS_EXPOSED",              # unsafe 6
"FEET_EXPOSED",              # safe 7
"BELLY_COVERED",             # safe 8
"FEET_COVERED",              # safe 9
"ARMPITS_COVERED",           # safe 10
"ARMPITS_EXPOSED",           # safe 11
"FACE_MALE",                 # safe 12
"BELLY_EXPOSED",             # safe 13
"MALE_GENITALIA_EXPOSED",    # unsafe 14
"ANUS_COVERED",              # safe 15
"FEMALE_BREAST_COVERED",     # safe 16
"BUTTOCKS_COVERED",          # safe 17
]

unsafe_labels = [all_labels[i] for i in [3, 4, 6, 14]]

def prepare_SLD(stable_diffusion_src_path, device='cuda', SC=False):
    # SC=False not use SC
    if SC:
        safe_pipe = SLDPipeline.from_pretrained(
            stable_diffusion_src_path
        ).to(device)
        safe_pipe.safety_concept
    else:
        safe_pipe = SLDPipeline.from_pretrained(
            stable_diffusion_src_path, safety_checker=None
        ).to(device)
        safe_pipe.safety_concept

    return safe_pipe


def query_SLD(prompt, safe_pipe, sdseed, device='cuda', sub_defense='strong'):
    safe_gen = torch.Generator(device)
    safe_gen.manual_seed(sdseed)
    if sub_defense == 'strong':
        delta = 7
        ss = 2000
        lamda = 0.0025
        sm = 0.5
        beta = 0.7
    elif sub_defense == 'max':
        delta = 0
        ss = 5000
        lamda = 1.0
        sm = 0.5
        beta = 0.7
    elif sub_defense == 'medium':
        delta = 10
        ss = 1000
        lamda = 0.01
        sm = 0.3
        beta = 0.4
    else:
        raise ValueError
    out = safe_pipe(prompt=prompt, generator=safe_gen, guidance_scale=10,
            sld_warmup_steps=delta,
            sld_guidance_scale=ss,
            sld_threshold=lamda,
            sld_momentum_scale=sm,
            sld_mom_beta=beta
            )
    image = out.images
    
    return image