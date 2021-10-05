from rest_framework import serializers
from .models import Tutorial
from django.contrib.auth.models import User

class TutorialSerializer(serializers.ModelSerializer):

    class Meta:
        model = Tutorial
        fields = (
            'id',
            'title',
            'description',
            'published'
        )
#  Basic user from tutorial
# class UserSerializer(serializers.HyperlinkedModelSerializer):
#     class Meta:
#         model = User
#         fields = ['url', 'username', 'email', 'groups']

# Revised user from tutorial
class UserSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'password']
        extra_kwargs = {'password': {'write_only': True, 'required': True}}

    # uses validated_data to make sure password is hashed
    def create(self, validated_data):
        user = User.objects.create_user(**validated_data)
        return user