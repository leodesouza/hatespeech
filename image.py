from PIL import Image
import os
import json


def show_image(file_name):
    img_txt_path = '/home/leonardosouza/projects/hatespeech_dataset/img_txt'
    img_txt_file_path = f'{img_txt_path}/{file_name}.json'
    if os.path.exists(img_txt_file_path):
        with open(img_txt_file_path, 'r') as file:
            text = json.load(file)['img_text']
            print(f'text on image: {text}')

    with open('/home/leonardosouza/projects/hatespeech_dataset/MMHS150K_GT.json', 'r') as file:
        _dict = json.load(file)
        for key in _dict:
            if key == file_name:
                info = _dict[key]
                tweet = info['tweet_text']
                print(f'text on tweet: {tweet}')

    img_txt_path = '/home/leonardosouza/projects/hatespeech_dataset/img_resized/'
    img_txt_path_file_name = img_txt_path + file_name + '.jpg'
    img = Image.open(img_txt_path_file_name)
    img.show()


if __name__ == '__main__':
    file_name = '1106733368547467264'
    show_image(file_name)
