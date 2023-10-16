from django.urls import path
from . import views

urlpatterns = [
    path('download/<str:video_filename>', views.download_video, name='video request handler'),
    path('register', views.client_register, name='client register handler'),
    path('exit', views.client_exit, name='client exit handler')
]
