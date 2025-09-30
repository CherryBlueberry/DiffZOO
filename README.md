# DiffZOO
Official implementation of the paper (NAACL 2025): DiffZOO: A Purely Query-Based Black-Box Attack for Red-Teaming Text-to-Image Generative Model via Zeroth Order Optimization

## Abstract
Current text-to-image (T2I) synthesis diffusion models raise misuse concerns, particularly in creating prohibited or not-safe-for-work (NSFW) images. To address this, various safety mechanisms and red teaming attack methods are proposed to enhance or expose the T2I model's capability to generate unsuitable content. However, many red teaming attack methods assume knowledge of the text encoders, limiting their practical usage. In this work, we rethink the case of purely black-box attacks without prior knowledge of the T2l model.  To overcome the unavailability of gradients and the inability to optimize attacks within a discrete prompt space, we propose DiffZOO which applies Zeroth Order Optimization to procure gradient approximations and harnesses both C-PRV and D-PRV to enhance attack prompts within the discrete prompt domain. We evaluated our method across multiple safety mechanisms of the T2I diffusion model and online servers. Experiments on multiple state-of-the-art safety mechanisms show that DiffZOO attains an 8.5% higher average attack success rate than previous works, hence its promise as a practical red teaming tool for T2l models.

## Setup: Create and Activate the Environment
Create a new Conda environment named `diffzoo` with Python 3.9 installed. Install all necessary packages listed in the `requirements.txt` file by running:

```bash
conda create -n diffzoo python=3.9
conda activate diffzoo
pip install -r requirements.txt
```

## Download Model
- Download BERT Model: [BERT Base Cased from Hugging Face](https://huggingface.co/google-bert/bert-base-cased/tree/main), and save the model files to the `./model/Bert` directory.
- Download CLIP Model: [CLIP from Hugging Face](https://huggingface.co/openai/clip-vit-large-patch14/tree/main), and save the model files to the `./models/CLIP/clip-vit-large-patch14` directory.
- Download stable-diffusion Model: [stable-diffusion from Hugging Face](https://huggingface.co/CompVis/stable-diffusion-v1-4/tree/main), and save the model files to the `./model/CompVis/stable-diffusion-v1-4` directory.

## Usage: Running DiffZOO
- To use the lightweight version of DiffZOO, known as DiffZOO-Lite, simply execute the following command:

```bash
python run_diffzoo.py --attack diffzoo-lite
```
- To use  DiffZOO, simply execute the following command:

```bash
python run_diffzoo.py --attack diffzoo
```
Results will be saved in the `results` folder.

## BibTeX
Please cite the paper:
Paper: https://arxiv.org/abs/2408.11071v2
```
@inproceedings{dang-etal-2025-diffzoo,
    title = "{D}iff{ZOO}: A Purely Query-Based Black-Box Attack for Red-teaming Text-to-Image Generative Model via Zeroth Order Optimization",
    author = "Dang, Pucheng  and
      Hu, Xing  and
      Li, Dong  and
      Zhang, Rui  and
      Guo, Qi  and
      Xu, Kaidi",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2025",
    year = "2025",
    pages = "17--31",
}
```
