from django.urls import path
from . import views

app_name = 'Ai_Hub'

urlpatterns = [
    path('', views.home, name='home'),
    path('tools/chat/', views.new_chat_view, name='chat'),
    path('tools/chat/<uuid:chat_id>/', views.chat_detail_view, name='chat_detail'),
    path('signup/', views.signup_view, name='signup'),
    path('verify/', views.verify_view, name='verify'),
    path('set-name/', views.set_name_view, name='set_name'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('tools/chat/<uuid:chat_id>/delete/', views.delete_chat_view, name='delete_chat'),
    path('tools/image/', views.image_generator, name='image_generator'),
    path('tools/prompt/', views.image_prompt, name='image_prompt'),
    path('tools/code/', views.code_writer, name='code_writer'),
    path('tools/resume/', views.resume_builder, name='resume_builder'),
    path('tools/essay/', views.essay_writer, name='essay_writer'),
    path('tools/grammar/', views.grammar_checker, name='grammar_checker'),
    path('tools/qr/', views.qr_generator, name='qr_generator'),
]