from django.db import models
from django.conf import settings
# Image Table uses djangos built in auto incremented id
class ImageTable(models.Model):
    """
    The ImageTable model represents the database table that stores records about images used
    for labeling by the users.

    Attributes:
    ----------
    pk: int
        Djangos built in auto incrementing primary key
    fileName: str
        The name of file that is with in the S3 bucket.
        Constraints: max_length=512 characters, must be unique, required
    imageUrl: str
        The full image url path to be provided to the front end for displaying the given image.
        Constraints: max_length=512 characters, must be unique, required
    label: boolean
        The ground truth label for a given image. True == healthy, False == unhealthy, default = false
    is_trainSet: boolean
        An indicator for the image to have the ability to be labeled by an user.
        If true then the image can be labeled by the user, otherwise the image is for the ML model.
    users: User
        The many-to-many relationship between images and users.
        Refer to Choices table for more information.
    """
    fileName = models.CharField(db_column='fileName', unique=True, max_length=512)  # unique column, logical PK
    imageUrl = models.CharField(db_column='imageUrl', unique=True, max_length=512)
    label = models.BooleanField(null=False)
    is_trainSet = models.BooleanField(default=True)
    users = models.ManyToManyField(settings.AUTH_USER_MODEL, through='Choice', related_name= 'choices_made')

    def __str__(self) -> str:
        return f'id: {self.pk}\n fileName: {self.fileName}\n imageUrl: {self.imageUrl}'

    class Meta:
        managed = True
        db_table = 'image_table' 

class Choice(models.Model):
    """
        The choices table is a historical record of a users labeling of images in a given session.
        This is taking a given snapshot of the labels of a set of images that are given to the ML model 
        to be trained on.

        Attributes:
        ----------
        user: User
            A foreign key to a given user who labeled a specific image.
        image: ImageTable
            A foreign key to a given image that was labeled by a user.
        userLabel: boolean
            The label given to a specific image by the user.
            True = healthy, False = unhealthy.
        create_date: DateTime
            The date a given choice record was created.
            Automatically given as the current date and time the record was created.
    """
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
    create_date = models.DateTimeField(auto_now_add=True)
    # TODO: add boolean field user_training_record, constraint = not null , front-end provides this

    def __str__(self) -> str:
        return f'ChoiceId: {self.pk} \n User: {self.user}\n Image: {self.image}\n Label: {self.userLabel}'

    class Meta:
        managed = True
        db_table = 'choice' 