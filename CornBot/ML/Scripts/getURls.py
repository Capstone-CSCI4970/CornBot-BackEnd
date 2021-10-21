"""
Created on 9/24/2021
@author: Bibek
"""
import pandas as pd
import os

def get_filenames_urls_labels():
    """
        Returns a tuple of lists of filenames,urls and labels in order.
        Returns
        -------
        returnValues : tuple
            Lists.
    """
    path = 's3://cornimagesbucket/csvOut.csv'# Path to the S3 bucket
    data = pd.read_csv(path, index_col = 0, header = None)#Read the csv
    data_temp = data.reset_index()#Recover the original index
    image_src = "cornimagesbucket.s3.us-east-2.amazonaws.com/images_compressed/"
    filenames = list(data_temp.iloc[:,0])#Get all the filename
    labels = list(data.iloc[:,-1].map(dict(B=1, H=0)))#Get corrosponding Labels of the filename
    file_urls = []
    for filename in filenames:
        file_urls.append(os.path.join(image_src,filename))#Src + filename is fileUrl
    return zip(filenames,file_urls,labels)

#How to Use it
# count = 0
# for i in get_filenames_urls_labels():
#     count += 1
#     print(i,'\n')
#     if count >= 5:
#         break


# Populate The Database
# id = 1
# for i in get_filenames_urls_labels():
#     new_entry = ImageTable(id,*i,True)
#     id += 1
#     new_entry.save()