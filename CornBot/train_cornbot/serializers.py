from .models import ImageTable
from rest_framework import serializers

class ImageTableSerializer(serializers.ModelSerializer):

    class Meta:
        model = ImageTable
        fields = (
            'imagename',
            'imageurl',
            'label'
        )