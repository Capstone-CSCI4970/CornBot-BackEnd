from django.conf.urls import url
from . import views
from django.urls import include, path
from rest_framework import routers
from rest_framework.authtoken.views import ObtainAuthToken

router = routers.DefaultRouter()
router.register(r'users', views.UserViewSet)

urlpatterns=[
    url(r'^', include(router.urls)),
    url(r'^auth/', ObtainAuthToken.as_view()),
    url(r'^images', views.image_list),
    url(r'^tester', views.test_choices),
    url(r'test/(?P<pk>[0-9]+)$', views.test_specific_choices),
    url(r'blag/(?P<pk>[0-9]+)$', views.get_user_choices_by_imageId)
]