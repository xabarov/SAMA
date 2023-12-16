import os

from utils.help_functions import is_im_path


def get_sim(model='all-distilroberta-v1', sentence1='I love to read books.',
            sentence2='Reading is a great way to relax.'):
    from sentence_transformers import SentenceTransformer, util

    if not os.path.exists(model):
        model = 'all-distilroberta-v1'

    roberta_model = SentenceTransformer(model)
    sentence1_embedding = roberta_model.encode(sentence1, convert_to_tensor=True)
    sentence2_embedding = roberta_model.encode(sentence2, convert_to_tensor=True)

    print(sentence1_embedding.shape)

    cosine_similarity_score = util.pytorch_cos_sim(sentence1_embedding, sentence2_embedding)

    return cosine_similarity_score.item()


def get_images_names_similarity(path, model='all-distilroberta-v1'):
    from sentence_transformers import SentenceTransformer, util

    if not os.path.exists(model):
        model = 'all-distilroberta-v1'

    roberta_model = SentenceTransformer(model)

    images = [im for im in os.listdir(path) if is_im_path(im)]

    emb = roberta_model.encode(images, convert_to_tensor=True)
    emb.to('cuda')
    cosine_similarity_score = util.pytorch_cos_sim(emb, emb)
    return cosine_similarity_score.detach().cpu().numpy()


if __name__ == '__main__':
    path = "D:\python\\aia_git\\ai_annotator\projects\\aes_test"
    sim = get_images_names_similarity(path,
                                      model="D:\\python\\aia_git\\ai_annotator\\sentence-transformers_all-distilroberta-v1")
    print(sim)
