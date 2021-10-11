from rest_framework import serializers
from django.contrib.auth.models import User
from .models import ImageTable, Choices

class UserSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'password']
        extra_kwargs = {'password': {'write_only': True, 'required': True}}

    # uses validated_data to make sure password is hashed
    def create(self, validated_data):
        user = User.objects.create_user(**validated_data)
        return user

class ImageSerializer(serializers.HyperlinkedModelSerializer):
    # users = UserSerializer(many = True)
    class Meta:
        model = ImageTable
        fields = (
            'fileName',
            'imageUrl',
            'label',
        )
class ChoiceSerializer(serializers.ModelSerializer):
    users = UserSerializer()
    images = ImageSerializer()
    class Meta:
        model = Choices
        fields = (
            'id',
            'users',
            'images',
            'userLabel'
        )
    depth = 1