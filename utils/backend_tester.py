import requests
import json

DOMEN_NAME = 'http://127.0.0.1:5000'


def test_detector():
    url = f'{DOMEN_NAME}/detect'
    files = {'file': open('image.jpg', 'rb')}
    response = requests.post(url, files=files, headers={'conf': '0.1', 'iou': '0.1', 'lrm': '0'})

    mask_results = json.loads(response.text)

    print(mask_results)


def test_sync_names():
    url = f'{DOMEN_NAME}/sync_names'
    response = requests.get(url)
    names = json.loads(response.text)
    print(names)


def encode(point_mass):
    result = []
    for p in point_mass:
        result.append([str(p[0]), str(p[1])])
    return result


def test_sam_box():
    input_box = [329, 767, 448, 897]

    url = f'{DOMEN_NAME}/sam_set_image'
    files = {'file': open('image2.jpg', 'rb')}

    is_set_need = True

    if is_set_need:
        res = requests.post(url, files=files)
        print(res.text)

    url = f'{DOMEN_NAME}/sam_box'

    sam_res = requests.post(url, json={'input_box': input_box})

    print(json.loads(sam_res.text))


def test_sam_points():
    points = [[437, 840],
              [382, 829]]
    labels = [0, 1]
    url = f'{DOMEN_NAME}/sam_set_image'
    files = {'file': open('image2.jpg', 'rb')}

    is_set_need = True

    if is_set_need:
        res = requests.post(url, files=files)
        print(res.text)

    points_json = points
    labels_json = labels

    url = f'{DOMEN_NAME}/sam_points'

    sam_res = requests.post(url, json={'input_points': points_json, 'input_labels': labels_json})

    print(json.loads(sam_res.text))


if __name__ == '__main__':
    test_sam_box()
