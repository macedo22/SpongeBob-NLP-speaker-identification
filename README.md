# SpongeBob Speaker Identification

The website can be found [here](https://whispering-harbor-58905.herokuapp.com/textclassifier/).

This simple web app takes either user-generated input text or direct quotes
from the cartoon, SpongeBob SquarePants, and predicts which character
among 10 possible characters was most likely to have said the quote. There
are three options for a user to choose from:

1) Have the classifier predict the speaker of the input text you provide,
   and it will compare the result to the "true speaker" that you provide.

2) Have the classifier predict the speaker of a random test quote by a
   character that you specify.
   
3) Have the classifier predict the speaker of a random test quote by a
   random character.

Quantitative comparative analysis is done between the predicted speaker and
the true speaker by comparying the weights of the n-grams present in the
input text for each, as well as the most commonly spoken n-grams by each
speaker.

If the application's guess is correct, then the comparative analysis will
instead be between the true speaker and the speaker who the application
believed was the 2nd most likely speaker of the input text.

Note that if the quote is user-generated, then the user also specifies which
character said the quote. In this case, the user's chosen speaker is considered
to be the "true speaker." Otherwise, a quote from the show is taken at random,
and the training label is the true speaker.
