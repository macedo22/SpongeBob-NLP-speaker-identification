from django import forms

class InputRandomForm(forms.Form): pass

class InputCharacterForm(forms.Form):
    speaker_choices = (('Anchovies', 'Anchovies'), ('DoodleBob', 'DoodleBob'),
                       ('Gary', 'Gary'), ('Mr. Krabs', 'Mr. Krabs'),
                       ('Narrator', 'Narrator'), ('Patrick', 'Patrick'),
                       ('Plankton', 'Plankton'), ('Sandy', 'Sandy'),
                       ('SpongeBob', 'SpongeBob'), ('Squidward', 'Squidward'))
    speaker = forms.ChoiceField(choices=speaker_choices)

class InputTextForm(forms.Form):
    input_text = forms.CharField(label='Input text', max_length=500,
                                 widget=forms.TextInput(attrs={'size':50}))
    speaker_choices = (('Anchovies', 'Anchovies'), ('DoodleBob', 'DoodleBob'),
                       ('Gary', 'Gary'), ('Mr. Krabs', 'Mr. Krabs'),
                       ('Narrator', 'Narrator'), ('Patrick', 'Patrick'),
                       ('Plankton', 'Plankton'), ('Sandy', 'Sandy'),
                       ('SpongeBob', 'SpongeBob'), ('Squidward', 'Squidward'))
    speaker = forms.ChoiceField(choices=speaker_choices)
