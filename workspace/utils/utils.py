import os
import requests
from requests.exceptions import HTTPError
import shutil
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import cv2


def load_df(csv_path):
    """
    Loads and transform Meli Datasets with some basic transformation.
    ---------
    csv_path (str): Path to the csv file.
    """

    schema={
        'item_id': str,
        'site_id': str,
        'domain_id': str,
        'picture_id': str,
        'correct_background?': 'category'
    }

    df=pd.read_csv(csv_path, dtype=schema)
    df['url']=df['picture_id'].apply(lambda x: (f'https://http2.mlstatic.com/D_{x}-F.jpg'))
    df['path']=df['picture_id'].apply(lambda x: (f'/workspace/data/production_imgs/D_{x}-F.jpg'))
    df=df.rename(columns={'correct_background?':'correct_background'})
    
    return df


def error_log(err, path):
    """
    Error logger.
    ---------
    err (error/Exception):  The error / Exception raised.
    path (str):             Path where to create the error log including filename.
    """
    with open(path, 'a') as log:
        log.write(err)



def get_pictures(picture_id):
    """
    Get as input the picture ID, formats it and downloads the pictures from MeLi API.
    Arguments
    ---------
    picture_id (str):  Id of the picture.
    """
    url = f'https://http2.mlstatic.com/D_{picture_id}-F.jpg'
    filename = url.split("/")[-1]
    not_available='img-not-available'
    not_found='https://http2.mlstatic.com/resources/frontend'

    try:
        response = requests.get(url, stream=True, timeout=(1, 4))
        # If the response was successful, no Exception will be raised
        response.raise_for_status()
        
    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
        log = (f'\n {http_err, url}, ')
        error_log(log, '/workspace/logs/error.log')

    except Exception as err:
        print(f'Other error occurred: {err}')
        log = (f'\n {err, url}, ')
        error_log(log, '/workspace/logs/error.log')

    else:
        # Check if the image was retrieved successfully
        if response.status_code == 200:
            # Check if server response is gif for not found
            splitted_url=response.url.split('/')
            if (not_available in splitted_url) or (not_found in response.url):
                print('Image Couldn\'t be retreived')

                img_log_path='/workspace/logs/img.log'
                new_log=(f'\n {url}, {response.url}, {response.status_code}, ')

                if os.path.exists(img_log_path):  
                    with open(img_log_path, 'a') as log:
                        log.write(new_log)
                else:
                    with open(img_log_path, 'w') as log:
                        header=f'request_url,response_url,status_code,extra'
                        log.write(header)
                        log.write(new_log)
            else:
                # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
                response.raw.decode_content = True
                # Open a local file with wb ( write binary ) permission.
                with open(filename,'wb') as f:
                    shutil.copyfileobj(response.raw, f)


def resize_image(im_path, root_path='./data/tf-imgs-resized/'):
    """
    Recieves and image and resize it for the input model format.
    if the image is not a square it creates black bands for the smaller size
    until it becomes a square.
    Arguments
    ---------
    im_path (str):  Path of the image to resize.
    root_path(str): Path of the directory where the resized images class directories will be.
    """
    try:
        receivedImage=Image.open(im_path)
        
        save_path=root_path + "/".join(im_path.split('/')[-2:])

        imageSize = receivedImage.size
        size=max(imageSize)
        x=imageSize[0]
        y=imageSize[1]

        # If image is not square, square it with black stripes
        if x > y:
            r = x - y
            yy = r // 2
            strip = Image.new('RGB', (x, yy), ('black'))
            squared = Image.new('RGB', (x, x))
            squared.paste(strip,(0,0))
            squared.paste(receivedImage,(0, yy))
            squared.paste(strip,(0, y+yy))
            if size > 320:
                im_resized=squared.resize((320,320))            
                im_resized.save(save_path)
            
            return(save_path)

        elif y > x:
            r = y - x
            xx = r // 2
            strip = Image.new('RGB', (xx, y), ('black'))
            squared = Image.new('RGB', (y, y))
            squared.paste(strip, (0, 0))
            squared.paste(receivedImage, (xx, 0))
            squared.paste(strip, (xx + x, 0))
            if size > 320:
                im_resized=squared.resize((320,320))            
                im_resized.save(save_path)
            
            return(save_path)

        elif(y==x):
            receivedImage=receivedImage.resize((320,320))
            receivedImage.save(save_path)
        
            return(save_path)

    except Exception as err:
        print(f'An error occurred: {err}')
        log = (f'\n {err, im_path}, ')
        error_log(log, '/workspace/logs/resize_imgs.log')



def tf_data(row, class_a, class_b):
    """
    Move images for tensorflow classification in one folder per class.
    Arguments
    ---------
    row (DataFrame row): Each row of a DataFrame. To be passed with the apply() method.
    class_a(str):       Path of the respective class folder in the file system.
    class_b(str):       Path of the respective class folder in the file system.
    """
    try:
        if row.correct_background == '1':
            path_a=class_a+'/'+row.path.split('/')[-1]
            shutil.move(row.path, path_a)
            return path_a

        elif row.correct_background == '0':
            path_b=class_b+'/'+row.path.split('/')[-1]
            shutil.move(row.path, path_b)
            return path_b

        else:
            return row.path
            
    except Exception as err:
        print(f'An error occurred: {err}')
        log = (f'\n {err, row}, ')
        error_log(log, '/workspace/logs/move.log')


def image_sizes(index: int, path: str):
    """
    Get image dimension and returns all the ones that matches RGB.
    Arguments
    ---------
    index (int): Image ID (Arrray index)
    path(str):   Image path in file system
    """
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
       
        if len(img.shape) != 3:
            log = (f'\nSHAPE!=3,{index},{path},{img.shape},')
            error_log(log, '/workspace/logs/img_size.log')
            print(f'Shape != 3 > {log}')
       
        shapes_list=[img.shape[0], img.shape[1], img.shape[2]]
        return shapes_list
        
    except Exception as err:
        print(f'An error occurred: {err}')
        log = (f'\n{err},{path},{index},')
        error_log(log, '/workspace/logs/img_size.log')
