from datetime import date, datetime
from time import timezone
from django.http import JsonResponse
from rest_framework import status
from rest_framework.parsers import JSONParser
from rest_framework.utils import json

from .serializers import ChoiceSerializer, UserSerializer, ImageSerializer
from .models import ImageTable, Choice
from django.contrib.auth.models import User
from rest_framework import viewsets
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import api_view, authentication_classes, permission_classes


from ML.ML_Class import ML_Model
from ML.Viz import make_confusion_matrix
from ML.analytics_viz import user_acc_viz,image_misclass_viz
from sklearn.ensemble import RandomForestClassifier
from ML.DataPreprocessing import DataPreprocessing
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import random
import torch
import cv2
from PIL import Image
from io import BytesIO
import base64
import matplotlib.pyplot as plt
from torchvision import transforms
import os

#Result Reproducibility
random.seed(7)

# Create your views here.
@api_view(['GET'])
@permission_classes([IsAuthenticated])
@authentication_classes([TokenAuthentication])
def get_images(request):
    """
        This function takes in a http GET request from the front-end to grab a random sample of 10 images 
        from the image table that are labeled to be training sets for the user(s) to label the given image
        in order to help train the ML model.

        Parameters
        ----------
        name : HttpRequest
            The incoming HttpRequest object used to determine what request is coming in.
        Returns
        ----------
        images: JsonResponse
            a json response containing a random list of images that the front-end will use
            to display to the user for labeling to help train the ML model.
    """
    images = ImageTable.objects.filter(is_trainSet=True)
    x = 10
    get_x_images_random = random.sample(list(images), x)#Get 'X' random Images 
    image_to_train = [it.pk for it in get_x_images_random]
    images = ImageTable.objects.filter(id__in=image_to_train)
    images_for_user = ImageSerializer(images, many=True)
    return JsonResponse(images_for_user.data, safe=False)



def get_data():
    """
        This Function uses the s3 bucket path to retrieve csv file containing the
        image names and its featutes.

        Returns
        ----------
        data: csv
            a csv file containing list of images and its features
    """
    path = 's3://cornimagesbucket/csvOut.csv'
    data = pd.read_csv(path, index_col = 0, header = None)
    return data


def retrive_prediction(train_img_names,train_img_label):
    """
        This function instantiates the ml_model class and retreives prediction
        of labels for train images

        Parameters
        ----------
        train_img_names : list
            It contains the list of image names being used for training
        
        train_img_label : list
            It contains the list of user assigned label for each image for training
        Returns
        ----------
        pred: list
            Prediction of image label for each image stored in a list 

    """
    data = get_data()
    train_set = data.loc[train_img_names, :]#get all the features of each images
    train_set['y_value'] = train_img_label#add a label column 
    ml_model = ML_Model(train_set, RandomForestClassifier(), DataPreprocessing(True))#instantiate the model
    pred = ml_model.predict_train_image()[0]#get the prediction
    return pred

def accuracy(x,y):
    """
        This function evaluate the accuracy based on ground truth and prediction. 
        It is represented in out of 100(percentages).

        Parameters
        ----------
        x : list
            It contains the list of goundtruth
        
        y : list
            It contains the list of prediction
        Returns
        ----------
        accuracy: int
            it returns the accurcy computed using x and y
    """
    if not np.array(x).any() or not np.array(y).any(): # if either list is empty, we cannot calculate the accuracy.
        return 0.00
    x,y = np.array(x),np.array(y)
    pred = (x == y).astype(np.int)
    accuracy = pred.mean()*100
    return accuracy

@api_view(['GET'])
@permission_classes([IsAuthenticated])
@authentication_classes([TokenAuthentication])
def get_acc(request,pk):
    """
        This function takes in a given request and user pk to get the
        users overally accuracy of labeling of the random set of images
        provided by get_images end-point and does this by comparing the
        users labels to the ground truth of the image.

        Parameters
        ----------
        request: HttpRequest
            The HttpRequest object received from the front-end.
        pk: int
            The identifier of a specific user to get only the accuracy for that user.
        data: JsonResponse
            it returns jsonrespone of data containing user_id, accuracy, and 
            all the images of user used to compute the accuracy.i.e images used
            for labelling for user 
    """
    user = User.objects.get(pk=pk)
    choices = Choice.objects.filter(user=user, user_training_record = False).order_by('image_id')
    imageid_choices = [choice.image_id for choice in choices]
    train_labels = [choice.userLabel for choice in choices]#User Train Labels
    images = ImageTable.objects.filter(pk__in=imageid_choices)
    train_images = [x.fileName for x in images]#User Train Images
    pred = retrive_prediction(train_images,train_labels)
    accuracy_train = accuracy(train_labels,pred)
    data = {'user_id':pk,'Accuracy':accuracy_train,'images':imageid_choices}
    return JsonResponse(data, safe=False)



@api_view(['GET'])
@permission_classes([IsAuthenticated])
@authentication_classes([TokenAuthentication])
def getTestAcc(request,pk):
    """
        This function takes in a given request and user pk to get the users
        overally accuracy of test set(the images not used for training).It
        computes the accuracy of the model on test set. Also, confidence for
        each images, and some other analytics like confusion matrix that are 
        helpful to better judge the model. 

        Parameters
        ----------
        request: HttpRequest
            The HttpRequest object received from the front-end.
        pk: int
            The identifier of a specific user to get only the accuracy for that user.
        data: JsonResponse
            it returns jsonrespone of data containing user_id, test accuracy, and 
            confidence for each images used for test accuracy. It also includes a 
            image that shows the confusion matrix and metrices score for the model. 
    """

    user = User.objects.get(pk=pk)

    item = request.GET.get('activity')
    choices = None
    if(item != None and item == '1'):
        choices = Choice.objects.filter(user=user, user_training_record = True).order_by('image_id')
    else:
        choices = Choice.objects.filter(user=user, user_training_record = False).order_by('image_id')
    
    ml_model = initialize_model(choices)
    data = get_data()
    images = ImageTable.objects.filter(is_trainSet=False)
    imageid_test = [img.imageUrl for img in images]
    test_labels = [x.label for x in images]#User Train Labels
    test_images = [x.fileName for x in images]#User Test Images
    test_set = data.loc[test_images, :]
    pred,prob = ml_model.predict_test_image(test_set)
    imgid_confid = zip(imageid_test,prob)
    model_image_confidence = []
    for image,conf in zip(imageid_test,prob):
        model_image_confidence.append(dict(imageUrl=image, confidence=conf))
    cf_matrix = confusion_matrix(test_labels,pred)
    labels = ['True Pos','False Neg','False Neg','True Neg']
    categories = ['Healthy', 'Blight']
    encoded_img = make_confusion_matrix(cf_matrix, group_names=labels, categories=categories, cmap='binary', title='Prediction CF Matrix',figsize=(12,12))
    confusion_matrix_uri = 'data:%s;base64,%s' % ('image/jpeg', encoded_img)
    accuracy_test = accuracy(test_labels,pred)
    data = {'user_id':pk,'Accuracy':accuracy_test,'image_confidence':list(model_image_confidence),'confusion_matrix_uri':confusion_matrix_uri}
    return JsonResponse(data, safe=False)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
@authentication_classes([TokenAuthentication])
def getUpload(request):
    """
        Endpoint uses yolov5 trained model to make inference in images

        Paramenters:
        ----------
        request: HttpRequest
            The HttpRequest object received from the front-end.
            
        Returns:
            URI of image returned by the object detection model
        ----------
    """
    file = request.FILES["uploadedFile"]
    #model = torch.hub.load('ML/yolov5', 'custom', path='ML/yolov5/runs/train/exp/weights/best.pt', source='local')
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='ML/best.pt')
    input = Image.open(file)
    transform = transforms.Compose([transforms.Resize(1056)])
    input = transform(input)
    results = model(input, size=1056)
    results.imgs # array of original images (as np array) passed to model for inference
    results.render()  # updates results.imgs with boxes and labels
    buffered = BytesIO()
    img_base64 = Image.fromarray(results.imgs[0])
    img_base64.save(buffered, format="JPEG")
    data = {"Pred_URI":base64.b64encode(buffered.getvalue()).decode('utf-8')}
    return JsonResponse(data, safe=False)

def initialize_model(choices):
    """
        Train the machine learning model based on the choices. 

        Paramenters:
        ----------
        choices: List
            choice reacord
            
        Returns:
            trained model
        ----------
    """
    if len(choices) == 0:
        return None 
    imageid_choices = [choice.image_id for choice in choices]
    train_labels = [choice.userLabel for choice in choices]#User Train Labels
    images_train = ImageTable.objects.filter(pk__in=imageid_choices)
    train_images = [x.fileName for x in images_train]#User Train Images
    data = get_data()
    label_image = zip(train_images,train_labels)
    label_image_unique = set(label_image)
    unzip_label_image_unique = zip(*label_image_unique)
    train_images,train_labels = unzip_label_image_unique
    train_set = data.loc[train_images, :]
    train_set['y_value'] = train_labels
    ml_model = ML_Model(train_set, RandomForestClassifier(), DataPreprocessing(True))
    return ml_model

def image_missclasfy_analytics():
    """
        Helper function to compute top five images that has been most frequently misclassified.

            
        Returns:
            List of top five misclassified images along number of times misclassified
        ----------
    """
    users = User.objects.all()
    anlytics_data = {}
    full_image_ids = np.zeros(len(ImageTable.objects.all())+1)
    for user in users:
        choices = Choice.objects.filter(user=user, user_training_record = False).order_by('image_id')
        if(len(choices) > 0):
            imageids = np.array([choice.image_id for choice in choices])
            user_labels = [choice.userLabel for choice in choices]
            images = ImageTable.objects.filter(pk__in=imageids)
            ground_truth = [x.label for x in images]
            mask = np.array(user_labels) != np.array(ground_truth)
            unmatched_labels = imageids[mask]
            full_image_ids[unmatched_labels] += 1
    
    id_misclas = np.argpartition(full_image_ids,-5)[-5:] # ids of 5 most misclassified image
    ids_value = full_image_ids[id_misclas]
    images_missclass = ImageTable.objects.filter(pk__in=id_misclas)
    id_to_imagename = [x.imageUrl for x in images_missclass]
    for idx,imageUrl in enumerate(id_to_imagename):
        anlytics_data[imageUrl] = ids_value[idx]
    return anlytics_data


def user_acc_analytics():
    """
        Helper function to compute user and their accuracy.
            
        Returns:
            List that includes all users and their repective accuracy
        ----------
    """
    users = User.objects.all()
    anlytics_data = {}

    for user in users:
        choices = Choice.objects.filter(user=user, user_training_record = False).order_by('image_id')
        ml_model = initialize_model(choices)
        if ml_model != None:
            data = get_data()
            images = ImageTable.objects.filter(is_trainSet=True)
            imageid_test = [img.id for img in images]
            test_labels = [x.label for x in images]#User Train Labels
            test_images = [x.fileName for x in images]#User Test Images
            test_set = data.loc[test_images, :]
            pred,prob = ml_model.predict_test_image(test_set)
            imgid_confid = zip(imageid_test,prob)
            anlytics_data[user.username] = round(accuracy(test_labels,pred),3)
        else:
            anlytics_data[user.username] = 0.0
    return anlytics_data

@api_view(['GET'])
@permission_classes([IsAuthenticated])
@authentication_classes([TokenAuthentication])
def get_user_acc_Analytics(request):
    """
        This end point shows test accuracy of user.

        Paramenters:
        ----------
        request: HttpRequest
            The HttpRequest mde by the front-end to this given end point
            
        Returns:
            JsonResponse that list of users and their test accuracy
        ----------
    """
    return JsonResponse(user_acc_analytics(), status=status.HTTP_200_OK)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
@authentication_classes([TokenAuthentication])
def get_misclasfy_image_Analytics(request):
    """
        This end point shows top five images that has been most frequently misclassified.

        Paramenters:
        ----------
        request: HttpRequest
            The HttpRequest mde by the front-end to this given end point
            
        Returns:
            JsonResponse list of top five misclassified images along number of times misclassified
        ----------
    """
    return JsonResponse(image_missclasfy_analytics(), status=status.HTTP_200_OK)
    
######## USER BASED VIEW #########

class UserViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """
    queryset = User.objects.all().order_by('-date_joined')
    serializer_class = UserSerializer

@api_view(['GET'])
@permission_classes([IsAuthenticated])
@authentication_classes([TokenAuthentication])
def image_list(request):
    """
        This is a simple API end-point that will return a list of all
        images to be displayed by the front-end
        Parameters
        ----------
        request: HttpRequest
            The HttpRequest object recieved in the GET request
        
        Returns
        ----------
        A JsonResponse with a list of all images for the front-end to display.
    """
    images = ImageTable.objects.filter(is_trainSet = True)
    if request.method == 'GET':
        image_serialized = ImageSerializer(images, many=True)
        return JsonResponse(image_serialized.data, safe=False)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
@authentication_classes([TokenAuthentication])
def test_choices(request):
    choices = Choice.objects.all()
    if request.method == 'GET':
        choice_seralizer = ChoiceSerializer(choices, many = True)
        return JsonResponse(choice_seralizer.data, safe=False)

# TODO: filter the users specific choices by todays date.
@api_view(['GET'])
@permission_classes([IsAuthenticated])
@authentication_classes([TokenAuthentication])
def get_user_specific_choices(request, pk):
    """
        This end-point takes a given request and user pk
        to get the users choices on images that were labeled during the current session.

        Parameters:
        ----------
        request: HttpRequest
            The HttpRequest that was made to the end point.
        pk: int
            The unique identifer for a given user.

        Returns:
        ----------
        JsonResponse with the set of choices a user has made.
    """

    user = User.objects.get(pk=pk)
    choices = Choice.objects.filter(user=user, user_training_record = False)
    serialized_data = ChoiceSerializer(choices, many=True)
    return JsonResponse(serialized_data.data, safe=False)

# TODO: filter the users specific choices by todays date.
@api_view(['GET'])
@permission_classes([IsAuthenticated])
@authentication_classes([TokenAuthentication])
def get_user_choices_by_imageId(request, pk):
    """
        This endpoint gets a users label on a specific image using
        the unique identifier provided and filters down by the current
        date.
    """
    images = ImageTable.objects.get(pk=pk)
    choices = Choice.objects.filter(image=images)

    data = ChoiceSerializer(choices, many = True)
    return JsonResponse(data.data, safe=False)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
@authentication_classes([TokenAuthentication])
def create_choice_record(request):
    """
        This end point creates choice record(s) that were made by
        user labeling images to help train the machine learning model.

        Parameters:
        ----------
        request: HttpRequest
            The HttpRequest made by the front-end to this given end point

        Returns:
        ----------
            JsonResponse returning the choice records made and http status of 201 Created.
    """
    choice_data = JSONParser().parse(request)
    choices_serializer = ChoiceSerializer(data=choice_data, many=True)
    if choices_serializer.is_valid():
        choices_serializer.save()
        return JsonResponse(choices_serializer.data, status=status.HTTP_201_CREATED, safe=False)
    else:
        return JsonResponse(choices_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['PUT'])
@permission_classes([IsAuthenticated])
@authentication_classes([TokenAuthentication])
def update_user_choice(request, pk):
    """
        This end point updates a given choice record by the pk provided in the request.

        Paramenters:
        ----------
        request: HttpRequest
            The HttpRequest mde by the front-end to this given end point
        pk: int
            The unique identifier of the choice record being updated.

        Returns:
            JsonResponse that includes the choice record that was updated.
        ----------
    """
    try:
        current_choice = Choice.objects.get(pk=pk)
    except Choice.DoesNotExist:
        return JsonResponse({'message': 'That choice does not exist'}, status=status.HTTP_404_NOT_FOUND)
    choice_data = JSONParser().parse(request)
    choice_seralizer = ChoiceSerializer(current_choice, data=choice_data)
    if choice_seralizer.is_valid():
        choice_seralizer.save()
        return JsonResponse(choice_seralizer.data)
    else:
        return JsonResponse(choice_seralizer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
@authentication_classes([TokenAuthentication])
def get_uid_by_username(request, user_name):
    """
        This end point takes a given user_name and finds the correlating user identifer 
        so that the front-end can save the identifier in the session to use for later
        information being requested about choices a given user makes.

        Parameters:
        ----------
        request: HttpRequest
            The HttpRequest object sent representing the request being made from the front-end.
        user_name: str
            The unique username to help identify the specifc user in the session to help retrieve
            the correlating unique identifier number otherwise know as the primary key.

        Returns:
        ----------
        JsonResponse with the correlating user_name and the unique identifier for that user.
    """
    try:
        user = User.objects.get(username=user_name)
    except User.DoesNotExist:
        return JsonResponse({'message': 'That user does not exist'}, status=status.HTTP_404_NOT_FOUND)
    uid = user.pk
    return JsonResponse({'username': user_name, 'uid': uid})

@api_view(['GET'])
@permission_classes([IsAuthenticated])
@authentication_classes([TokenAuthentication])
def users_accuracy_leaderboard(request):
    """
        This end point shows leaderboard based on user's accuracy.

        Paramenters:
        ----------
        request: HttpRequest
            The HttpRequest mde by the front-end to this given end point
            
        Returns:
            JsonResponse that includes all users and their repective accuracy
        ----------
    """
    users = User.objects.all()
    data = {}

    for user in users:
        data[user.username] = {}
        choices = Choice.objects.filter(user=user, user_training_record = False).order_by('image_id')
        imageid_choices = [choice.image_id for choice in choices]
        train_labels = [choice.userLabel for choice in choices]#User Train Labels
        groundTruths = ImageTable.objects.filter(pk__in=imageid_choices).order_by('id')
        choices = sorted(choices, key= lambda item: (item.image.id, item.create_date)) # ordering to ensure we get the most recent labeling incase same image labeld twice
        my_dict = dict()
        for choice in choices:
            my_dict[choice.image_id] = choice.userLabel
        u_labels = list()
        for key in sorted(my_dict):
                u_labels.append(my_dict[key])
        user_image_truths = [image.label for image in groundTruths]
        data[user.username] = accuracy(u_labels, user_image_truths) #if user_choices != None  else 0
    return JsonResponse(data, status=status.HTTP_200_OK)

def get_filenames_urls_labels():
    """
        Gets filename of each images, along with urls and its ground truth label from
        csv file stored in aws S3

        Returns:
            JsonResponse zipped filenames, file urls, and labels 
        ----------
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
# Endpoint to populate image table with 20 or 200 images set for test
# TODO Either lock down this endpoint or improve solution such that this is not needed.
@api_view(['GET'])
def image_populate(request):
    """
        This end point fills up the ImageTable table. It sets 80% data as train set and 
        reserves rest of 20% for test set.

        Paramenters:
        ----------
        request: HttpRequest
            The HttpRequest mde by the front-end to this given end point

        Returns:
            JsonResponse indicating sucessfull population along with total images populated
        ----------
    """
    images = list(get_filenames_urls_labels())
    healthy_test_images = images[0:10]
    unhealthy_test_images = images[-10:]
    images_for_users = images[10:-10]
    #insert healthy images for model testing
    saveImage(healthy_test_images, False)
    #insert unhealth images for model testing
    saveImage(unhealthy_test_images, False)
    #insert images for users to label
    saveImage(images_for_users, True)
    totalImages = ImageTable.objects.all().count()
    return JsonResponse({'message': 'Image Table populated', 'Total images': totalImages}, status=status.HTTP_200_OK)

# save the given image list with the is_trainSet indicator
def saveImage(image_list, is_trainSet):
    """
        This end point is helper function for populate image that saves each image to
        the tables

        Paramenters:
        ----------
        image_list: List
            List of images
        
        is_trainSet: boolean
            True if the image is being used for training
        ----------
    """
    for image in image_list:
        new_entry = ImageTable( fileName=image[0], imageUrl=image[1], label =image[2], is_trainSet=is_trainSet)
        new_entry.save()
        
