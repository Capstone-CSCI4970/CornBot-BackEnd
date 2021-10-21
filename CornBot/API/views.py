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


import pandas as pd
import random
# Create your views here.

#Result Reproducibility
random.seed(7)
# Create your views here.
@api_view(['GET'])
def get_images(request):
    if request.method == 'GET':
        #Reset
        #ImageTable.objects.all().update(is_train=0)
        images = ImageTable.objects.filter(is_trainSet=True)
        x = 10
        get_x_images_random = random.sample(list(images), x) 
        image_to_train = [it.pk for it in get_x_images_random]
        images = ImageTable.objects.filter(id__in=image_to_train)
        images.update(is_trainSet=False)
        all_healthy_serializer = ImageSerializer(images, many=True)
        return JsonResponse(all_healthy_serializer.data, safe=False)
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