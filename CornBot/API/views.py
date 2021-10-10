from django.http import JsonResponse
from rest_framework import status
from rest_framework.parsers import JSONParser

from .serializers import UserSerializer

from django.contrib.auth.models import User
from rest_framework import viewsets
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated
# Create your views here.

######## USER BASED VIEW #########

class UserViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """
    queryset = User.objects.all().order_by('-date_joined')
    serializer_class = UserSerializer

