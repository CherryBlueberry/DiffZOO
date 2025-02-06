import torch
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from PIL import Image
import pickle


def load_prompts(file_path, device):
    return torch.FloatTensor(pickle.load(open(file_path, 'rb'))).to(device)

class Q16():
    def __init__(self, device):
        file_path = './model/q16/prompts.p'
        self.torch_device = device
        self.safety_prompts = load_prompts(file_path, device)

        self.model = CLIPVisionModelWithProjection.from_pretrained("./models/CLIP/clip-vit-large-patch14").to(self.torch_device)
        self.processor = CLIPImageProcessor.from_pretrained("./models/CLIP/clip-vit-large-patch14")

    def q16_classifier(self, embeddings, verbose=False):
        safety_prompts_norm = self.safety_prompts / self.safety_prompts.norm(dim=-1, keepdim=True)
        image_features_norm = embeddings / embeddings.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features_norm @ safety_prompts_norm.T)
        # values, indices = similarity[0].topk(5)#
        probs = similarity.squeeze().softmax(dim=-1)
        if verbose:
            print(probs)
        prediction_score, pred_label_idx = torch.topk(probs.float(), 1)
        return pred_label_idx.squeeze()

    @torch.no_grad()
    def detect(self, image: Image):
        clip_input = self.processor(images=image, return_tensors="pt").to(self.torch_device)
        image_embeds = self.model(clip_input.pixel_values).image_embeds
        q16_safety_classfier_res = self.q16_classifier(image_embeds)
        unsafe_res = [bool(res) for res in q16_safety_classfier_res]

    @torch.no_grad()
    def q16_sim(self, x, image):
        clip_x = self.processor(images=x, return_tensors="pt").to(self.torch_device)
        x_embeds = self.model(clip_x.pixel_values).image_embeds

        clip_image = self.processor(images=image, return_tensors="pt").to(self.torch_device)
        image_embeds = self.model(clip_image.pixel_values).image_embeds

        x_features_norm = x_embeds / x_embeds.norm(dim=-1, keepdim=True)
        image_features_norm = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        similarity = (100.0 * x_features_norm @ image_features_norm.T)
        return similarity

    @torch.no_grad()
    def q16_prob(self, x):
        clip_x = self.processor(images=x, return_tensors="pt").to(self.torch_device)
        x_embeds = self.model(clip_x.pixel_values).image_embeds
        x_features_norm = x_embeds / x_embeds.norm(dim=-1, keepdim=True)
        safety_prompts_norm = self.safety_prompts / self.safety_prompts.norm(dim=-1, keepdim=True)
        similarity = (100.0 * x_features_norm @ safety_prompts_norm.T)

        probs = similarity.squeeze().softmax(dim=-1)
        # prediction_score, pred_label_idx = torch.topk(probs.float(), 1)

        return probs
    
    @torch.no_grad()
    def q16_prob_list(self, x_list):
        prob_list = []
        for x in x_list:
            clip_x = self.processor(images=x, return_tensors="pt").to(self.torch_device)
            x_embeds = self.model(clip_x.pixel_values).image_embeds
            x_features_norm = x_embeds / x_embeds.norm(dim=-1, keepdim=True)
            safety_prompts_norm = self.safety_prompts / self.safety_prompts.norm(dim=-1, keepdim=True)
            similarity = (100.0 * x_features_norm @ safety_prompts_norm.T)

            probs = similarity.squeeze().softmax(dim=-1)
            # prediction_score, pred_label_idx = torch.topk(probs.float(), 1)
            prob_list.append(probs)

        return prob_list