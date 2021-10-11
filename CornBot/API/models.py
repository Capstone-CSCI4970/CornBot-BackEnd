from django.db import models
from django.conf import settings
# Image Table uses djangos built in auto incremented id
class ImageTable(models.Model):
    fileName = models.CharField(db_column='fileName', unique=True, max_length=512)  # unique column, logical PK
    imageUrl = models.CharField(db_column='imageUrl', unique=True, max_length=512)
    label = models.BooleanField(null=False)
    is_trainSet = models.BooleanField(default=True)
    users = models.ManyToManyField(settings.AUTH_USER_MODEL, through='Choices', related_name= 'choices_made')

    class Meta:
        managed = True
        db_table = 'image_table'

class Choices(models.Model):
    users = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null= True
        )
    images = models.ForeignKey(
        ImageTable,
        on_delete=models.SET_NULL,
        null=True
    )
    userLabel = models.BooleanField()
    confidence = models.DecimalField(default=0.00, max_digits=5, decimal_places=2)