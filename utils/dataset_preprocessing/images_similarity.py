import os

from PIL import Image
from transformers import AutoProcessor, CLIPModel

from utils.help_functions import is_im_path
from tqdm import tqdm
import torch
from ui.signals_and_slots import LoadPercentConnection

conn = LoadPercentConnection()


def image_preprocess(path):
    base_width = 640
    img = Image.open(path)
    wpercent = (base_width / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((base_width, hsize), Image.Resampling.LANCZOS)
    return img


def print_percent(value):
    print(value)


def get_images_similarity(path, model_path="openai/clip-vit-large-patch14", percent_hook=None):
    from sentence_transformers import util

    if not os.path.exists(model_path):
        model_path = "openai/clip-vit-large-patch14"

    images = [im for im in os.listdir(path) if is_im_path(im)]

    if percent_hook:
        conn.percent.connect(percent_hook)

    model = CLIPModel.from_pretrained(model_path)
    if torch.cuda.is_available():
        model.to('cuda')

    # Get the image features
    processor = AutoProcessor.from_pretrained(model_path)
    print("start preprocessing images...")
    images_pil = []
    for i in tqdm(range(len(images))):
        filename = images[i]
        full_name = os.path.join(path, filename)
        images_pil.append(image_preprocess(full_name))
        if percent_hook:
            conn.percent.emit(int(50 * i / len(images)))

    print("create embeddings...")
    embeddings = []
    for i in tqdm(range(len(images_pil))):
        image = images_pil[i]
        input = processor(images=image, return_tensors="pt")
        if torch.cuda.is_available():
            input.to('cuda')
        emb = model.get_image_features(**input)
        embeddings.append(emb.detach().cpu().numpy().tolist()[0])
        if percent_hook:
            conn.percent.emit(50 + int(50.0 * i / len(images_pil)))

    emb = torch.tensor(embeddings)
    if torch.cuda.is_available():
        emb.to('cuda')
    print("calc cos similarities...")
    cosine_similarity_score = util.pytorch_cos_sim(emb, emb)
    return cosine_similarity_score.detach().cpu().numpy()


if __name__ == '__main__':
    path = "D:\python\\aia_git\\ai_annotator\\test_projects\\aes_500"
    sim = get_images_similarity(path,
                                model_path="openai/clip-vit-base-patch32", percent_hook=print_percent)
    print(sim)
