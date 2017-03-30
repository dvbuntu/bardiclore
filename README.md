# Shakespearean Text Generation

This is "Bardic Lore", an exploration of generating Shakespearean looking text with recurrent neural networks.  Certainly this kind of thing has been done before; this is learning by doing.

All code is in Keras 2.0 using Shakespeare downloaded from [Project Gutenberg](http://www.gutenberg.org/ebooks/100.txt.utf-8).  This includes all his plays and sonnets.  To model and generate text, I lowercased everything, and stripped out extraneous punctuation.

# Inspirations

* [Keras LSTM Example](https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py)
* [Char-RNN](https://github.com/karpathy/char-rnn)
* [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

# Example generated text

After training all night, the model starts to sound Middle Englishy:

> r els entreat me to learn me hath the death to make with a man to can the head;
>  englon came our duke of gentle wood of my house with thy sends of lading and dumpers of the stanf
>  than so much that the world of hot stand a prince;
>  but i am to do stand, and i have the enemy of the life and thou doth but you do the stand of the fortunes of soul to conferer.
>  my lord, and the garter are or to the pardon of the graces as so, and he would not the man, when they are good nor and your or others
> personal use only, and are but the while of my gracious bottom of the true man,
>  i will you shall for the son of the horses.
>  but i shall see him be a fair duke of the which i should to arm'd it of the deadful honest of the are 
>  may i will did the stans bowes a sort 
>  the rage as the prince, that i will see the capalous lords.
>  but in the stand the complete bear a world with like the sent of the morting sail. 
>  who should come to my lords of the prince to deserve the stay of the lady 
>  but the more heart


