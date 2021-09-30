from django.http import JsonResponse
from rest_framework import status
from rest_framework.parsers import JSONParser

from .serializers import TutorialSerializer, UserSerializer
from .models import Tutorial

from django.contrib.auth.models import User
from rest_framework import viewsets
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated


# Create your views here.
from rest_framework.decorators import api_view, permission_classes, authentication_classes


@api_view(['GET', 'POST', 'DELETE'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def tutorial_list(request):
    # Get list of tutorials, post a new tutorial, delete all tutorials
    if request.method == 'GET':
        tutorials = Tutorial.objects.all()
        title = request.query_params.get('title', None)
        if title is not None:
            tutorials = tutorials.filter(title__icontains=title)

        tutorials_serializer = TutorialSerializer(tutorials, many=True)
        return JsonResponse(tutorials_serializer.data, safe=False)
        # safe=False for objects serialization
    elif request.method == 'POST':
        tutorial_data = JSONParser().parse(request)
        tutorials_serializer = TutorialSerializer(data=tutorial_data)
        if tutorials_serializer.is_valid():
            tutorials_serializer.save()
            return JsonResponse(tutorials_serializer.data, status=status.HTTP_201_CREATED)
        return JsonResponse(tutorials_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    elif request.method == 'DELETE':
        count = Tutorial.objects.all().delete()
        return JsonResponse({'message': '{} tutorials were deleted'.format(count[0])}, status=status.HTTP_204_NO_CONTENT)

@api_view(['GET', 'PUT', 'DELETE'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def tutorial_detail(request, pk):
    # find the tutorial by pk
    try:
        tutorial = Tutorial.objects.get(pk=pk)
    except Tutorial.DoesNotExist:
        return JsonResponse({'message': 'The tutorial does not exist'}, status=status.HTTP_404_NOT_FOUND)

    if request.method == 'GET':
        tutorial_serializer = TutorialSerializer(tutorial)
        return JsonResponse(tutorial_serializer.data)

    elif request.method == 'PUT':
        tutorial_data = JSONParser().parse(request)
        tutorial_serializer = TutorialSerializer(tutorial, data=tutorial_data)
        if tutorial_serializer.is_valid():
            tutorial_serializer.save()
            return JsonResponse(tutorial_serializer.data)
        return JsonResponse(tutorial_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    elif request.method == 'DELETE':
        tutorial.delete()
        return JsonResponse ( {'message': 'Tutorial was deleted'},  status=status.HTTP_204_NO_CONTENT)

@api_view(['GET'])
def tutorial_list_published(request):
    # GET published tutorials
    tutorials = Tutorial.objects.filter(published=True)

    if request.method == 'GET':
        tutorial_serializer = TutorialSerializer(tutorials, many=True)
        return JsonResponse(tutorial_serializer.data, safe=False)


######## USER BASED VIEW #########

class UserViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """
    queryset = User.objects.all().order_by('-date_joined')
    serializer_class = UserSerializer

