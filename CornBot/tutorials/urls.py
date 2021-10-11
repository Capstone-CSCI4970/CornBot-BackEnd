from django.conf.urls import url
from . import views
from django.urls import include, path
from rest_framework import routers
from rest_framework.authtoken.views import ObtainAuthToken


urlpatterns = [
    url(r'^tutorials$', views.tutorial_list),
    url(r'^tutorials/(?P<pk>[0-9]+)$', views.tutorial_detail),
    url(r'^tutorials/published$', views.tutorial_list_published)
]

# # Wire up our API using automatic URL routing.
# # Additionally, we include login URLs for the browsable API.
# urlpatterns = [
# ]