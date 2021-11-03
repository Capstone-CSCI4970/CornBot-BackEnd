import json

from django.contrib.auth.models import User
from django.urls import reverse

from rest_framework import status
from rest_framework.authtoken.models import Token
from rest_framework.test import APITestCase

from tutorials.models import Tutorial
from tutorials.serializers import TutorialSerializer


class RegistrationTestCase(APITestCase):

    def test_registration(self):

        print("Testing user registration")
        data = {
            "username": "testcase",
            "email" : "testcase@email.com",
            "password": "testcasepassword"
        }
        response = self.client.post("/api/users/" , data)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED) # example of test passing
        # self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT) # example of test failing

