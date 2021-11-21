from rest_framework import serializers
from django.contrib.auth.models import User
from .models import ImageTable, Choice

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
    class Meta:
        model = ImageTable
        fields = (
            'id',
            'fileName',
            'imageUrl',
            'label',
        )
class ChoiceSerializer(serializers.ModelSerializer):

    class Meta:
        model = Choice
        fields = (
            'id',
            'user',
            'image',
            'userLabel',
            'user_training_record'
        )
    def create(self, validated_data):
        # get the associated user and image
        user = validated_data.pop('user')
        image = validated_data.pop('image')
        choice_object = Choice.objects.create(user=user, image=image, userLabel = validated_data.pop('userLabel'),
             user_training_record =validated_data.pop('user_training_record'))
        return choice_object

    def update(self, instance, validated_data):
        instance.userLabel = validated_data.get('userLabel', instance.userLabel)
        instance.save()
        return instance     