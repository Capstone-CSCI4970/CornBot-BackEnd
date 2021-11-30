import json

from django.contrib.auth.models import User
from django.urls import reverse

from rest_framework import status
from rest_framework.authtoken.models import Token
from rest_framework.test import APITestCase
from .models import Choice, ImageTable

from django.urls import get_resolver

class ApiTest(APITestCase):

    users = {
        'TestCase1': {
            "username": "TestCase1",
            "email" : "testcase1@email.com",
            "password": "testcasepassword"
        },
        'TestCase2': {
            "username": "TestCase2",
            "email" : "testcase2@email.com",
            "password": "testcasepassword"
        },
        'TestCase3': {
            "username": "TestCase3",
            "email" : "testcase3@email.com",
            "password": "testcasepassword"
        }
    }

    choices = {
    }

    Images = {
      'Image1':  {
            "id": 1,
            "fileName": "DSC00160.JPG",
            "imageUrl": "cornimagesbucket.s3.us-east-2.amazonaws.com/images_compressed/DSC00160.JPG",
            "label": True
        },
      'Image2':  {
            "id": 2,
            "fileName": "DSC00167.JPG",
            "imageUrl": "cornimagesbucket.s3.us-east-2.amazonaws.com/images_compressed/DSC00167.JPG",
            "label": True
        },
       'Image3': {
            "id": 3,
            "fileName": "DSC00207.JPG",
            "imageUrl": "cornimagesbucket.s3.us-east-2.amazonaws.com/images_compressed/DSC00207.JPG",
            "label": True
        },
       'Image4': {
        "id": 4,
        "fileName": "DSC00040.JPG",
        "imageUrl": "cornimagesbucket.s3.us-east-2.amazonaws.com/images_compressed/DSC00040.JPG",
        "label": False
        },
       'Image5': {
           "id": 5,
            "fileName": "DSC00047.JPG",
            "imageUrl": "cornimagesbucket.s3.us-east-2.amazonaws.com/images_compressed/DSC00047.JPG",
            "label": False
        },
       'Image6': {
            "id": 6,
            "fileName": "DSC00264.JPG",
            "imageUrl": "cornimagesbucket.s3.us-east-2.amazonaws.com/images_compressed/DSC00264.JPG",
            "label": False
        }
    }
    # Set up test database with test data.
    def setUp(self) -> None:
        # Create users for testing
        for user in self.users:
            User.objects.create_user(user)
        # Create images for testing
        for image in self.Images.values():
           item =  ImageTable(**image)
           item.save()
        # set up token credentials for "logged in" user
        self.admin = User.objects.get(username=self.users['TestCase1']['username'])
        token = Token.objects.create(user = self.admin )
        self.client.credentials(HTTP_AUTHORIZATION='Token ' + token.key)

    def logger(self, log):
        print(f'\tLog [INFO]: {log}\n')
    
    # Get user id of a valid user
    def test_getUserId_success(self):
        self.logger('test_getUserId_success')
        url = reverse('getId', kwargs={'user_name': self.admin})
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
    
    def test_getUserId_fail(self):
        self.logger('test_getUserId_fail')
        url = reverse('getId', kwargs={'user_name': 'invalid_user'})
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_create_choice(self):
        self.logger('test_create_choice')
        self.logger(f'User pk: {self.admin.pk}')
        choice = [
            {
                "user": self.admin.pk,
                "image": 1,
                "userLabel": True,
                "user_training_record": True
            }
        ]
        url = reverse('createChoice')
        self.logger(f'This is : {url}')
        response = self.client.post(url, choice, format = 'json')
        # Test response code, total objects, user who created, image associated
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(Choice.objects.count(), 1)
        self.assertEqual(Choice.objects.get().user, self.admin)
        self.assertEqual(Choice.objects.get().image, ImageTable.objects.get(pk=1))


    def test_registration(self):
        self.logger('test_registration')
        newUser = {
            "username": "TestCase5",
            "email" : "testcase5@email.com",
            "password": "testcasepassword"
        }

        print("Testing successful user registration")
        response = self.client.post("/api/users/" , newUser)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED) # example of test passing
        # self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT) # example of test failing

