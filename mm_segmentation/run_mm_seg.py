from mmseg.apis import inference_model, init_model, show_result_pyplot
import mmcv

from utils import cls_settings

classes = cls_settings.CLASSES_SEG
palette = cls_settings.PALETTE_SEG

def run_seg(cnn_seg_type, image_file_name, save_file_name, device='cuda:0', opacity=0.5):

    config, checkpoint = cls_settings.get_cfg_and_weights_by_cnn_seg_name(cnn_seg_type)

    model = init_model(config, checkpoint, device=device)

    result = inference_model(model, image_file_name)

    show_result_pyplot(model, image_file_name, result, out_file=save_file_name, opacity=opacity)

if __name__ == '__main__':
    cnn_seg_type = 'PSPNet'
    image_file_name = 'bel_doel_2016_4.jpg'
    save_name = 'results.jpg'
    run_seg(cnn_seg_type, image_file_name, save_name)