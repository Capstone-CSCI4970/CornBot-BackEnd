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
