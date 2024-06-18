# My First Vtuber project

This repo is an attemp to build an AI vtuber by open source resource.

This project is of course inspired by the famous AI vtuber Neuro-sama and her dad Vedal.

But the infrastructure is 100% designed on my own.

Below is a brief introduction of My First Vtuver


# Design

A human could interact with world, consider, and take action. So do an AI vtuber.

Currently, My first Vtuber have only ear, but no eye. It could here from audio input of microphone.

After it here something,  it will interpret the audio signal as text.

Then the text will provide input of brain, which is a LLM model. 

And the brain will output some text to express its intention.

The intention forms several actions. 

First, it like to express through facial expression, which by Live2D here to control the character.

Second, it like to express throuth talking. The text is been feed into BARK model to produce voice.

Above is the basic flow of My First Vtuber.

We could interact with My First Vtuber by talking to it, get response from it and repeat this loop.


