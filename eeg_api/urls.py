from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('models/', views.models_view, name='models'),
    path('predict/', views.predict_view, name='predict'),
]
