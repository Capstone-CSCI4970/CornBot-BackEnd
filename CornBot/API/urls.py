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
    url(r'^getuid/(?P<user_name>\w+)/$', views.get_uid_by_username, name='getId'),
    url(r'^images', views.image_list),
    url(r'^getimages', views.get_images),
    url(r'^getacc/(?P<pk>[0-9]+)$', views.get_acc),
    url(r'^getTestAcc/(?P<pk>[0-9]+)$', views.getTestAcc),
    url(r'choice/(?P<pk>[0-9]+)$', views.get_user_specific_choices),
    url(r'user/choice/(?P<pk>[0-9]+)$', views.get_user_choices_by_imageId),
    url(r'^choice/create', views.create_choice_record, name='createChoice'),
    url(r'^choice/(?P<pk>[0-9]+)$', views.update_user_choice),
    url(r'^get_users_leaderboard', views.users_accuracy_leaderboard)
]