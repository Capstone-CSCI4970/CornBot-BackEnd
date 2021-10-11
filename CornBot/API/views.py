from django.http import JsonResponse
from rest_framework import status
from rest_framework.parsers import JSONParser

from .serializers import ChoiceSerializer, UserSerializer, ImageSerializer
from .models import ImageTable, Choices
from django.contrib.auth.models import User
from rest_framework import viewsets
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import api_view
# Create your views here.

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
    choices = Choices.objects.all()
    if request.method == 'GET':
        choice_seralizer = ChoiceSerializer(choices, many = True)
        return JsonResponse(choice_seralizer.data, safe=False)

@api_view(['GET'])
def test_specific_choices(request, pk):
    user = User.objects.get(pk=pk)
    choices = Choices.objects.filter(users=user)

    serialized_data = ChoiceSerializer(choices, many=True)
    return JsonResponse(serialized_data.data, safe=False)

@api_view(['GET'])
def get_user_choices_by_imageId(request, pk):
    images = ImageTable.objects.get(pk=pk)
    choices = Choices.objects.filter(images=images)

    data = ChoiceSerializer(choices, many = True)
    return JsonResponse(data.data, safe=False)
