from django.http import JsonResponse
from rest_framework import status
from rest_framework.parsers import JSONParser

from .serializers import ChoiceSerializer, UserSerializer, ImageSerializer
from .models import ImageTable, Choice
from django.contrib.auth.models import User
from rest_framework import viewsets
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import api_view


from ML.ML_Class import ML_Model
from ML.Viz import make_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from ML.DataPreprocessing import DataPreprocessing
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import random
# Create your views here.

#Result Reproducibility
random.seed(7)
# Create your views here.
@api_view(['GET'])
def get_images(request):
    #Reset
    #ImageTable.objects.all().update(is_trainSet=True)
    images = ImageTable.objects.filter(is_trainSet=True)
    x = 10
    get_x_images_random = random.sample(list(images), x)#Get 'X' random Images 
    image_to_train = [it.pk for it in get_x_images_random]
    images = ImageTable.objects.filter(id__in=image_to_train)
    images.update(is_trainSet=False)# Is used for training
    all_healthy_serializer = ImageSerializer(images, many=True)
    return JsonResponse(all_healthy_serializer.data, safe=False)



def get_data():
    path = 's3://cornimagesbucket/csvOut.csv'
    data = pd.read_csv(path, index_col = 0, header = None)
    return data

def retrive_prediction(train_img_names,train_img_label):
    data = get_data()
    train_set = data.loc[train_img_names, :]
    train_set['y_value'] = train_img_label
    ml_model = ML_Model(train_set, RandomForestClassifier(), DataPreprocessing(True))
    return ml_model.predict_train_image()[0]

def accuracy(x,y):
    x,y = np.array(x),np.array(y)
    pred = (x == y).astype(np.int)
    return pred.mean()*100

@api_view(['GET'])
def get_acc(request,pk):
    user = User.objects.get(pk=pk)
    choices = Choice.objects.filter(user=user)
    imageid_choices = [choice.image_id for choice in choices]
    train_labels = [choice.userLabel for choice in choices]#User Train Labels
    images = ImageTable.objects.filter(pk__in=imageid_choices)
    train_images = [x.fileName for x in images]#User Train Images
    pred = retrive_prediction(train_images,train_labels)
    # cf_matrix = confusion_matrix(train_labels,pred)
    # labels = ['True Neg','False Pos','False Neg','True Pos']
    # categories = ['Healthy', 'Blight']
    # encoded_img = make_confusion_matrix(cf_matrix, group_names=labels, categories=categories, cmap='binary', title='Prediction CF Matrix',figsize=(12,12))
    # image_uri = 'data:%s;base64,%s' % ('image/jpeg', encoded_img)
    accuracy_train = accuracy(train_labels,pred)
    data = {'user_id':pk,'Accuracy':accuracy_train,'images':imageid_choices}
    # data = {'user_id':pk,'Accuracy':accuracy_train,'images':imageid_choices,'image_uri':image_uri}
    return JsonResponse(data, safe=False)
    

def retrive_prediction(train_img_names,train_img_label):
    data = get_data()
    train_set = data.loc[train_img_names, :]
    train_set['y_value'] = train_img_label
    ml_model = ML_Model(train_set, RandomForestClassifier(), DataPreprocessing(True))
    return ml_model.predict_train_image()[0]

@api_view(['GET'])
def getTestAcc(request,pk):
    user = User.objects.get(pk=pk)
    
    choices = Choice.objects.filter(user=user)
    imageid_choices = [choice.image_id for choice in choices]
    train_labels = [choice.userLabel for choice in choices]#User Train Labels
    images_train = ImageTable.objects.filter(pk__in=imageid_choices)
    train_images = [x.fileName for x in images_train]#User Train Images
    data = get_data()
    train_set = data.loc[train_images, :]
    train_set['y_value'] = train_labels
    ml_model = ML_Model(train_set, RandomForestClassifier(), DataPreprocessing(True))

    images = ImageTable.objects.filter(is_trainSet=True)
    imageid_test = [img.id for img in images]
    test_labels = [x.label for x in images]#User Train Labels
    test_images = [x.fileName for x in images]#User Test Images
    test_set = data.loc[test_images, :]
    pred,prob = ml_model.predict_test_image(test_set)
    print("###########")
    print(type(prob),prob)
    imgid_confid = zip(imageid_test,prob)

    cf_matrix = confusion_matrix(test_labels,pred)
    labels = ['True Neg','False Pos','False Neg','True Pos']
    categories = ['Healthy', 'Blight']
    encoded_img = make_confusion_matrix(cf_matrix, group_names=labels, categories=categories, cmap='binary', title='Prediction CF Matrix',figsize=(12,12))
    confusion_matrix_uri = 'data:%s;base64,%s' % ('image/jpeg', encoded_img)
    accuracy_test = accuracy(test_labels,pred)
    data = {'user_id':pk,'Accuracy':accuracy_test,'image_confidence':list(imgid_confid),'confusion_matrix_uri':confusion_matrix_uri}
    return JsonResponse(data, safe=False)

######## USER BASED VIEW #########

class UserViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """
    queryset = User.objects.all().order_by('-date_joined')
    serializer_class = UserSerializer

@api_view(['GET'])
def image_list(request):
    images = ImageTable.objects.all()
    if request.method == 'GET':
        image_serialized = ImageSerializer(images, many=True)
        return JsonResponse(image_serialized.data, safe=False)

@api_view(['GET'])
def test_choices(request):
    choices = Choice.objects.all()
    if request.method == 'GET':
        choice_seralizer = ChoiceSerializer(choices, many = True)
        return JsonResponse(choice_seralizer.data, safe=False)

@api_view(['GET'])
def test_specific_choices(request, pk):
    user = User.objects.get(pk=pk)
    choices = Choice.objects.filter(user=user)

    serialized_data = ChoiceSerializer(choices, many=True)
    return JsonResponse(serialized_data.data, safe=False)

@api_view(['GET'])
def get_user_choices_by_imageId(request, pk):
    images = ImageTable.objects.get(pk=pk)
    choices = Choice.objects.filter(image=images)

    data = ChoiceSerializer(choices, many = True)
    return JsonResponse(data.data, safe=False)

@api_view(['POST'])
def create_choice_record(request):
    choice_data = JSONParser().parse(request)
    choices_serializer = ChoiceSerializer(data=choice_data)
    if choices_serializer.is_valid():
        choices_serializer.save()
        return JsonResponse(choices_serializer.data, status=status.HTTP_201_CREATED, safe=False)
    else:
        return JsonResponse(choices_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['PUT'])
def update_user_choice(request, pk):
    try:
        current_choice = Choice.objects.get(pk=pk)
    except Choice.DoesNotExist:
        return JsonResponse({'message': 'That choice does not exist'}, status=status.HTTP_404_NOT_FOUND)
    
    print(f'this is the current choice grabbed: {current_choice}')
    choice_data = JSONParser().parse(request)
    choice_seralizer = ChoiceSerializer(current_choice, data=choice_data)
    if choice_seralizer.is_valid():
        choice_seralizer.save()
        return JsonResponse(choice_seralizer.data)
    else:
        return JsonResponse(choice_seralizer.errors, status=status.HTTP_400_BAD_REQUEST)