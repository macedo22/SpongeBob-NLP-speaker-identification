from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render, get_object_or_404
from django.urls import reverse
from django.template import loader
from django.views import generic

from .forms import InputTextForm, InputCharacterForm, InputRandomForm
from . import runner

class IndexView(generic.ListView):
    template_name = 'textclassifier/index.html'

    def get_queryset(self):
        return None


class DetailView(generic.DetailView):
    template_name = 'textclassifier/detail.html'


class ResultsView(generic.DetailView):
    template_name = 'textclassifier/results.html'


def make_results_page(request, input_text):
    return render(request, 'textclassifier/results.html', {'input_text': input_text})


def get_results(request):
    if request.method == 'POST':
        input_info = request.POST.get("input_info", None)
        if input_info == "text":
            form = InputTextForm(request.POST)
        elif input_info == "character":
            form = InputCharacterForm(request.POST)
        else:
            form = InputRandomForm(request.POST)

        if form.is_valid():
            model_stuff = runner.setup()
            sentiment = model_stuff[0]
            cls = model_stuff[1]

            if input_info == "text":
                input_text = form.cleaned_data['input_text']
                speaker = form.cleaned_data['speaker']
                context = runner.run_given_both(sentiment, cls, input_text, speaker)

            elif input_info == "character":
                speaker =  form.cleaned_data['speaker']
                context = runner.run_given_label(sentiment, cls, speaker)
            else:
                context = runner.run_random(sentiment, cls)

            return render(request, 'textclassifier/results.html', context)

    else:
        input_text_form = InputTextForm()
        input_character_form = InputCharacterForm()
        input_random_form = InputRandomForm()

    return render(request, 'textclassifier/index.html',
                  {'input_text_form': input_text_form,
                   'input_character_form': input_character_form,
                   'input_random_form': input_random_form})
