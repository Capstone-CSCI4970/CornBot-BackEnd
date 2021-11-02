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
    url(r'^getuid/(?P<uname>\w+)/$', views.get_uid_by_username),
    url(r'^images', views.image_list),
    url(r'^getimages', views.get_images),
    url(r'^getacc/(?P<pk>[0-9]+)$', views.get_acc),
    url(r'^tester', views.test_choices),
    url(r'choice/(?P<pk>[0-9]+)$', views.test_specific_choices),
    url(r'user/choice/(?P<pk>[0-9]+)$', views.get_user_choices_by_imageId),
    url(r'^choice/create', views.create_choice_record),
    url(r'^choice/(?P<pk>[0-9]+)$', views.update_user_choice)
]