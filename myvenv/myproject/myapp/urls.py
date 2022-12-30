from unicodedata import name
from django.contrib import admin
from django.urls import path,include
from . import views
urlpatterns = [
    path('',views.index,),
    path('predict/',views.predict,name='predictor'),
    path('predict/result',views.result)
]
