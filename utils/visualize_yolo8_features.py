from ultralytics import YOLO


def visualize_features(yolo_model, image_path, conf=0.5):
    yolo_model.predict(image_path, save=True, conf=conf, visualize=True)


if __name__ == '__main__':
    aes_model_path = "../yolov8/weights/best.pt"
    aes_image = "../projects/aes_last/germany_gundremmingen_4.jpg"
    model = YOLO(aes_model_path)

    visualize_features(model, aes_image, conf=0.5)
