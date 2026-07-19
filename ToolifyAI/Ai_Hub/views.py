from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib.auth import login, logout, authenticate
from django.contrib import messages
from django.core.mail import send_mail
from django.conf import settings
from django.http import JsonResponse
from django.db.models import Count
from datetime import datetime
from groq import Groq
from huggingface_hub import InferenceClient

import io
import base64
import qrcode

from .predict import route_to_module
from .forms import SignUpForm
from .models import EmailVerification, Chat, ChatMessage

client = Groq(api_key=settings.GROQ_API_KEY)
hf_client = InferenceClient(api_key=settings.HF_API_KEY)


def get_greeting():
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning"
    elif hour < 17:
        return "Good afternoon"
    return "Good evening"


def get_username(request):
    """Small helper so every view doesn't repeat this line."""
    return request.user.first_name or request.user.email.split('@')[0]


def ask_ai(prompt: str) -> str:
    """
    Same pattern as chat_detail_view's Groq call, just reusable for
    single-shot tools that don't need conversation history.
    """
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error: {e}"


@login_required(login_url='Ai_Hub:login')
def home(request):
    if request.method == "POST":
        user_input = request.POST.get("user_input")
        # Stash the message so the destination tool can auto-send it
        # instead of making the user retype what they already wrote here.
        request.session['pending_message'] = user_input
        target_url = route_to_module(user_input)
        return redirect(target_url)

    context = {
        "greeting": get_greeting(),
        "username": get_username(request),
    }
    return render(request, "Ai_Hub/home.html", context)


@login_required(login_url='Ai_Hub:login')
def new_chat_view(request):
    chat = Chat.objects.create(user=request.user)
    return redirect('Ai_Hub:chat_detail', chat_id=chat.id)


@login_required(login_url='Ai_Hub:login')
def chat_detail_view(request, chat_id):
    chat = get_object_or_404(Chat, id=chat_id, user=request.user)

    if request.method == "POST":
        user_message = request.POST.get('message', '').strip()
        ai_text = ""
        if user_message:
            ChatMessage.objects.create(chat=chat, role='user', text=user_message)

            if not chat.title:
                chat.title = user_message[:50]
                chat.save()

            api_messages = []
            for msg in chat.messages.all():
                role = "assistant" if msg.role == "ai" else "user"
                api_messages.append({"role": role, "content": msg.text})

            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=api_messages,
                )
                ai_text = response.choices[0].message.content
            except Exception as e:
                ai_text = f"Something went wrong: {e}"

            ChatMessage.objects.create(chat=chat, role='ai', text=ai_text)

        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({"reply": ai_text})

        return redirect('Ai_Hub:chat_detail', chat_id=chat.id)

    all_chats = Chat.objects.filter(user=request.user).annotate(
        msg_count=Count('messages')
    ).filter(msg_count__gt=0)

    return render(request, "Ai_Hub/chat.html", {
        "chat": chat,
        "history": chat.messages.all(),
        "all_chats": all_chats,
        "greeting": get_greeting(),
        "username": get_username(request),
        "prefill_message": request.session.pop('pending_message', ''),
    })


@login_required(login_url='Ai_Hub:login')
def delete_chat_view(request, chat_id):
    chat = get_object_or_404(Chat, id=chat_id, user=request.user)

    if request.method == "POST":
        chat.delete()

    return redirect('Ai_Hub:home')


def signup_view(request):
    if request.method == "POST":
        form = SignUpForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data['email']
            password = form.cleaned_data['password']

            user = User.objects.create_user(username=email, email=email, password=password)
            user.is_active = False
            user.save()

            code = EmailVerification.generate_code()
            EmailVerification.objects.create(user=user, code=code)

            send_mail(
                subject="Your ToolifyAI verification code",
                message=f"Your verification code is: {code}\nIt expires in 10 minutes.",
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=[email],
            )

            request.session['pending_user_id'] = user.id
            return redirect('Ai_Hub:verify')
    else:
        form = SignUpForm()

    return render(request, "Ai_Hub/signup.html", {"form": form})


def verify_view(request):
    user_id = request.session.get('pending_user_id')
    if not user_id:
        return redirect('Ai_Hub:signup')

    if request.method == "POST":
        entered_code = request.POST.get('code', '').strip()

        try:
            user = User.objects.get(id=user_id)
            verification = EmailVerification.objects.get(user=user)
        except (User.DoesNotExist, EmailVerification.DoesNotExist):
            return render(request, "Ai_Hub/verify_failed.html", {
                "reason": "Something went wrong. Please sign up again."
            })

        if verification.is_expired():
            user.delete()
            del request.session['pending_user_id']
            return render(request, "Ai_Hub/verify_failed.html", {
                "reason": "That code expired. Please sign up again."
            })

        if entered_code == verification.code:
            user.is_active = True
            user.save()
            verification.delete()
            del request.session['pending_user_id']
            login(request, user, backend='django.contrib.auth.backends.ModelBackend')
            return redirect('Ai_Hub:set_name')
        else:
            return render(request, "Ai_Hub/verify_failed.html", {
                "reason": "That code was incorrect."
            })

    return render(request, "Ai_Hub/verify.html")


@login_required(login_url='Ai_Hub:login')
def set_name_view(request):
    if request.method == "POST":
        name = request.POST.get('name', '').strip()
        if name:
            request.user.first_name = name
            request.user.save()
        return redirect('Ai_Hub:home')

    return render(request, "Ai_Hub/set_name.html")


def login_view(request):
    if request.method == "POST":
        email = request.POST.get('email')
        password = request.POST.get('password')
        user = authenticate(request, username=email, password=password)
        if user is not None:
            login(request, user)
            return redirect('Ai_Hub:home')
        else:
            messages.error(request, "Invalid email or password.")

    return render(request, "Ai_Hub/login.html")


def logout_view(request):
    logout(request)
    return redirect('Ai_Hub:login')


@login_required(login_url='Ai_Hub:login')
def image_generator(request):
    if request.method == "POST" and request.headers.get("X-Requested-With") == "XMLHttpRequest":
        prompt = request.POST.get("message", "")

        try:
            # The SDK figures out the correct provider/routing internally --
            # this is why we're using it instead of a raw URL we'd have to
            # guess and maintain ourselves.
            pil_image = hf_client.text_to_image(
                prompt,
                model="black-forest-labs/FLUX.1-schnell",
            )

            # pil_image is a Pillow Image object -- encode it to base64
            # the same way we do for the QR generator.
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
            image_url = f"data:image/png;base64,{encoded}"
            return JsonResponse({"image_url": image_url})

        except Exception as e:
            print(f"🔍 Image generation failed: {e}")
            return JsonResponse(
                {"error": "Image generation failed. Please try again in a moment."},
                status=502,
            )

    return render(request, "Ai_Hub/image_generator.html", {
        "username": get_username(request),
        "prefill_message": request.session.pop('pending_message', ''),
    })


@login_required(login_url='Ai_Hub:login')
def image_prompt(request):
    if request.method == "POST" and request.headers.get("X-Requested-With") == "XMLHttpRequest":
        idea = request.POST.get("message", "")
        prompt = (
            "Turn this short idea into a single, detailed, vivid "
            "AI image-generation prompt (2-3 sentences, no preamble): "
            f"{idea}"
        )
        reply = ask_ai(prompt)
        return JsonResponse({"reply": reply})
    return render(request, "Ai_Hub/image_prompt.html", {
        "username": get_username(request),
        "prefill_message": request.session.pop('pending_message', ''),
    })


@login_required(login_url='Ai_Hub:login')
def code_writer(request):
    if request.method == "POST" and request.headers.get("X-Requested-With") == "XMLHttpRequest":
        message = request.POST.get("message", "")
        prompt = (
            "Write code for this request. First give a 1-2 sentence explanation "
            "in plain text, then provide the complete code inside a single "
            "markdown code block with the correct language tag, like ```python. "
            "Do not put any explanation text inside the code block itself. "
            f"Request: {message}"
        )
        reply = ask_ai(prompt)
        return JsonResponse({"reply": reply})
    return render(request, "Ai_Hub/code_writer.html", {
        "username": get_username(request),
        "prefill_message": request.session.pop('pending_message', ''),
    })


@login_required(login_url='Ai_Hub:login')
def resume_builder(request):
    if request.method == "POST" and request.headers.get("X-Requested-With") == "XMLHttpRequest":
        message = request.POST.get("message", "")
        prompt = (
            "You are a professional resume writer. Based on this "
            "background, write a polished resume section (bullet points, "
            f"action verbs, no preamble): {message}"
        )
        reply = ask_ai(prompt)
        return JsonResponse({"reply": reply})
    return render(request, "Ai_Hub/resume_builder.html", {
        "username": get_username(request),
        "prefill_message": request.session.pop('pending_message', ''),
    })


@login_required(login_url='Ai_Hub:login')
def essay_writer(request):
    if request.method == "POST" and request.headers.get("X-Requested-With") == "XMLHttpRequest":
        message = request.POST.get("message", "")
        prompt = f"Write a well-structured essay on the following topic: {message}"
        reply = ask_ai(prompt)
        return JsonResponse({"reply": reply})
    return render(request, "Ai_Hub/essay_writer.html", {
        "username": get_username(request),
        "prefill_message": request.session.pop('pending_message', ''),
    })


@login_required(login_url='Ai_Hub:login')
def grammar_checker(request):
    if request.method == "POST" and request.headers.get("X-Requested-With") == "XMLHttpRequest":
        message = request.POST.get("message", "")
        prompt = (
            "Correct the grammar and spelling in this text. Return only "
            f"the corrected version, no explanation: {message}"
        )
        reply = ask_ai(prompt)
        return JsonResponse({"reply": reply})
    return render(request, "Ai_Hub/grammar_checker.html", {
        "username": get_username(request),
        "prefill_message": request.session.pop('pending_message', ''),
    })


@login_required(login_url='Ai_Hub:login')
def qr_generator(request):
    if request.method == "POST" and request.headers.get("X-Requested-With") == "XMLHttpRequest":
        text = request.POST.get("message", "")

        # Generated fully offline with the qrcode library -- no external
        # API, no key, and it can never go down or rate-limit you.
        qr_image = qrcode.make(text)
        buffer = io.BytesIO()
        qr_image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # A data: URL lets the browser display the image directly --
        # no need to save a file to disk or set up MEDIA_URL for this.
        image_url = f"data:image/png;base64,{encoded}"
        return JsonResponse({"image_url": image_url})
    return render(request, "Ai_Hub/qr_generator.html", {
        "username": get_username(request),
        "prefill_message": request.session.pop('pending_message', ''),
    })