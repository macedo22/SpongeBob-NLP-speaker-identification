from django.urls import path

from . import views

app_name = 'textclassifier'
urlpatterns = [
    path('', views.get_results, name='index'),
    path('<int:pk>/', views.DetailView.as_view(), name='detail'),
    path('results/', views.ResultsView.as_view(), name='results'),
]
