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


if __name__ == '__main__':
    test_detector()
