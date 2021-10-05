from django.shortcuts import render
from .models import ImageTable
from .serializers import ImageTableSerializer
from django.http import JsonResponse
from rest_framework.decorators import api_view
import os

import random

#Result Reproducibility
random.seed(7)
# Create your views here.
@api_view(['GET'])
def image_list(request):
    if request.method == 'GET':
        #Reset
        #ImageTable.objects.all().update(is_train=0)


        images = ImageTable.objects.filter(is_train=0)
        x = 10
        get_x_images_random = random.sample(list(images), x) 
        image_to_train = [it.pk for it in get_x_images_random]
        images = ImageTable.objects.filter(imagename__in=image_to_train)
        images.update(is_train=1)
        all_healthy_serializer = ImageTableSerializer(images, many=True)
        return JsonResponse(all_healthy_serializer.data, safe=False)

