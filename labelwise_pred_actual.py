import pandas as pd
import numpy as np
import cv2
import shutil
import os
from tqdm import tqdm

def plot_rec(coor, img,flag, label=''):
    """
    This Function plots the annoations on the images along with label.
    Args:
    1.coor: --tuple. The coornidates of the  bbox, it should have the data in the following format (xmin,ymin,xmax,ymax).
    2.image: --np array. The image object containing the image in np.array must be provided.
    3.label: -- str. The label for the bbox to be mentioned here.

    Returns:
    The image with the annotaions and label written on the image.
    """
    if flag:
        #black is the predicted output and if work on predicted annotation.
        x1, y1, x2, y2 = coor
        draw_img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), thickness=3)
        cv2.putText(draw_img, label, (int(x1), int(y1)), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)
    else:
        #blue is the actual output and else work on actual annotation.
        x1, y1, x2, y2 = coor
        draw_img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), thickness=3)
        cv2.putText(draw_img, label, (int(x1), int(y1)), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
    return draw_img

def plot_pred_actual_labelwise(actual_csv_path,predicted_csv_path, annotated_files_out_folder_path, original_images_input_folder_path, first_5_only=False):
    """
    This Function plots the annotations on the images along with label and saves it labelwise.
    Args:
    1.actual_csv_path: --str. The path to the actual test.csv, it should have the data in the following format (path,xmin,ymin,xmax,ymax,label).
    2.predicted_csv_path: --str. The path to the predicted.csv, it should have the data in the following format (path,xmin,ymin,xmax,ymax,label).
    3.annotated_files_out_folder_path: --str. The path to directory where the new annotated images will be saved.
    4.original_images_input_folder_path: --str. The path to images directory.
    5.first_5_only: --Boolean. Default: False This parameter by default will allow for plotting of all the annotations.
    Change it to True to plot only 5 images per label.
    
    Returns:
    Label wise images are plotted with the annotaions and labels and stored in the folder mentioned.
    """
    act_data_df = pd.read_csv(actual_csv_path)
    pred_data_df = pd.read_csv(predicted_csv_path)
    A_lable_list = set(act_data_df.label)

    for i in tqdm(A_lable_list, desc='Processing labels for all images.'):
        path = os.path.join(annotated_files_out_folder_path, 'labelwise_annotations', str(i))
        if not os.path.exists(path):
            os.makedirs(path)
        if first_5_only:
            Act_temp_df = act_data_df.loc[act_data_df['label'] == i].head()
            Pred_temp_df = pred_data_df.loc[pred_data_df['label'] == i].head()
        else:
            Act_temp_df = act_data_df.loc[act_data_df['label'] == i]
            Pred_temp_df = pred_data_df.loc[pred_data_df['label'] == i]
        if len(Act_temp_df) > 0:
            page_list = set(Act_temp_df.path)
            for k in page_list:
                A_image_temp_df = Act_temp_df.loc[Act_temp_df['path'] == k]
                P_image_temp_df = Pred_temp_df.loc[Pred_temp_df['path'] == k]

                image_path = os.path.join(original_images_input_folder_path, str(k))
                img = cv2.imread(image_path)
                for j, t in A_image_temp_df.iterrows():
                    x1 = t.xmin
                    y1 = t.ymin
                    x2 = t.xmax
                    y2 = t.ymax
                    label = str(t.label)
                    flag=False
                    anno_image = plot_rec((x1, y1, x2, y2), img, flag, label)
                for c, d in P_image_temp_df.iterrows():
                    x1 = d.xmin
                    y1 = d.ymin
                    x2 = d.xmax
                    y2 = d.ymax
                    label = str(d.label)
                    flag=True
                    anno_image = plot_rec((x1, y1, x2, y2), anno_image, flag, label)
                cv2.imwrite(os.path.join(path, str(k)), anno_image)


#function calling
actual_csv_path='../test.csv'
predicted_csv_path='../predicted.csv'
original_images_input_folder_path='../test_images/'
annotated_files_out_folder_path='../folder_name/'

plot_pred_actual_labelwise(actual_csv_path,predicted_csv_path,original_images_input_folder_path,annotated_files_out_folder_path)
