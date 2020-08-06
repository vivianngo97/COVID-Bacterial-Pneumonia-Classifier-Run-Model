# COVID and Bacterial Pneumonia Detector

In this repo, we train a deep learning model to distinguish between x-ray images of healthy individuals and those with COVID-19 or bacterial pneumonia.


# Test it out

To test out this punctuation restoration model, you just need to run [__Models.py__](https://github.com/vivianngo97/Punctuation_Transcription/blob/master/play.py). Please follow the steps below. Note that you may be required to install some modules using __requirements.txt__.

## Via Command Line:
- Clone this repository COVID-19-X-ray-Classifier
- Navigate to the directory : COVID-19-X-ray-Classifier/Run_Model
- Type __pip install -r requirements.txt__ to install the requirements from requirements.txt
- Type __python Models.py__
- You can now test out the model by following the prompts!
- You will see something like this: 

<pre><code>>>> Here are the possible files to pick from:
['healthy4.jpeg', 'bac2.jpeg', 'bac3.jpeg', 'healthy2.jpeg', 'bac4.jpeg', 'healthy3.jpeg', 'healthy1.jpeg', 'covid4.png', 'covid1.jpg', 'bac1.jpeg', 'covid3.png', 'covid2.png'] 

>>> Please enter the file name of your image of interest (please include extension):

>>> ~~~~~~ Our prediction is:  <>  ~~~~~~

>>> Would you like to play again? If yes, type y or Y:

>>> Have a nice day!

</code></pre>

- If you are curious, you may also add your own image to COVID-19-X-ray-Classifier/Run_Model/Sample_Images and predict the classes of those.
