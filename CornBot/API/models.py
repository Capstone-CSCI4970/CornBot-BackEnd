from django.db import models
from django.conf import settings
# Image Table uses djangos built in auto incremented id
class ImageTable(models.Model):
    fileName = models.CharField(db_column='fileName', unique=True, max_length=512)  # unique column, logical PK
    imageUrl = models.CharField(db_column='imageUrl', unique=True, max_length=512)
    label = models.BooleanField(null=False)
    is_trainSet = models.BooleanField(default=True)
    users = models.ManyToManyField(settings.AUTH_USER_MODEL, through='Choice', related_name= 'choices_made')

    class Meta:
        managed = True
        db_table = 'image_table' 

    def __str__(self) -> str:
        return f'id: {self.pk}\n fileName: {self.fileName}\n imageUrl: {self.imageUrl}'

class Choice(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        null= True
        )
    image = models.ForeignKey(
        ImageTable,
        on_delete=models.CASCADE,
        null=True
    )
    userLabel = models.BooleanField()

    def __str__(self) -> str:
        return f'ChoiceId: {self.pk} \n User: {self.user}\n Image: {self.image}\n Label: {self.userLabel}'