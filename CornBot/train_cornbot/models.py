from django.db import models

# Create your models here.
class ImageTable(models.Model):
    imagename = models.CharField(db_column='ImageName', primary_key=True, max_length=512)  # Field name made lowercase.
    imageurl = models.CharField(db_column='ImageUrl', unique=True, max_length=512)  # Field name made lowercase.
    label = models.IntegerField()
    is_train = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'image_table'