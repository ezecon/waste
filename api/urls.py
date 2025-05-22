from django.urls import path
from .views import ImageClassificationView

urlpatterns = [
    path('upload/', ImageClassificationView.as_view(), name='photo-upload'),
]
