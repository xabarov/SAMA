from mmseg.apis import inference_model, init_model, show_result_pyplot
from mmseg.apis import MMSegInferencer


def segment(image_path, config_path, checkpoint_path, palette, classes, device='cuda'):
    # config_path = '../work_dirs/psp_aes/psp_aes.py'
    # checkpoint_path = '../work_dirs/psp_aes/iter_5000.pth'
    # palette = [[0, 0, 0], [255, 0, 0], [0, 255, 0]]
    # classes = [
    #     'background', 'water', 'vapor'
    # ]

    inferencer = MMSegInferencer(model=config_path,
                                 palette=palette,
                                 classes=classes,
                                 weights=checkpoint_path,
                                 device=device)
    result = inferencer(image_path, show=False)

    return result


def list_of_models():
    # models is a list of model names, and them will print automatically
    print(MMSegInferencer.list_models('mmseg'))


def init_model_demo():
    config_path = '../work_dirs/psp_aes/psp_aes.py'
    checkpoint_path = '../work_dirs/psp_aes/iter_6000.pth'

    # initialize model without checkpoint
    # model = init_model(config_path)

    # init model and load checkpoint
    model = init_model(config_path, checkpoint_path)

    # init model and load checkpoint on CPU
    # model = init_model(config_path, checkpoint_path, 'cpu')

    img_path = 'austria_zwentendorf_4.jpg'

    result = inference_model(model, img_path)

    # print(result)
    vis_image = show_result_pyplot(model, img_path, result, show=True, out_file='work_dirs/result.png')
    print(vis_image)


if __name__ == '__main__':
    config = "../mm_segmentation/configs/psp_aes.py"
    checkpoint = "../mm_segmentation/checkpoints/iter_52000_83_59.pth"
    im = 'D:\python\datasets\\aes_2200_copy\\argentina_atucha_1.jpg'
    palette = [[0, 0, 0], [255, 0, 0], [0, 255, 0]]
    classes = [
        'background', 'water', 'vapor'
    ]
    results = segment(image_path=im, config_path=config, checkpoint_path=checkpoint, palette=palette, classes=classes,
                      device='cuda')

    print(results)
