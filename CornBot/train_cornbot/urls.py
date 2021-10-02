from django.conf.urls import url
from . import views


urlpatterns=[
    url(r'^viewimage/$',views.image_list),
]