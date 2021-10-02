from django.shortcuts import render
from .models import ImageTable
from .serializers import ImageTableSerializer
from django.http import JsonResponse
from rest_framework.decorators import api_view

# Create your views here.
@api_view(['GET'])
def image_list(request):
    # Get list of tutorials, post a new tutorial, delete all tutorials
    if request.method == 'GET':
        images = ImageTable.objects.all()[:5]
        image_serializer = ImageTableSerializer(images, many=True)
        return JsonResponse(image_serializer.data, safe=False)
        # safe=False for objects serialization