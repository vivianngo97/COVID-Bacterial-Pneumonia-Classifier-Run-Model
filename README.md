# COVID and Bacterial Pneumonia Detector

In [__COVID-19-X-ray-Classifier__](https://github.com/vivianngo97/COVID-19-X-ray-Classifier), we train a deep learning model to distinguish between x-ray images of healthy individuals and those with COVID-19 or bacterial pneumonia. Because that repo includes all training and testing images, the repo is very large. In order to make experimentation easier for the public, we built this smaller repo. 

If you would like to run the model, please use this repository. If you would like to learn about the model training process or read our paper, please visit [COVID-19-X-ray-Classifier](https://github.com/vivianngo97/COVID-19-X-ray-Classifier).


# Test it out

To test out this classification model, you just need to run [__Models.py__](https://github.com/vivianngo97/COVID-Bacterial-Pneumonia-Classifier-Run-Model/blob/master/Run_Model/Models.py). Please follow the steps below. Note that you may be required to install some modules using [__requirements.txt__](https://github.com/vivianngo97/COVID-Bacterial-Pneumonia-Classifier-Run-Model/blob/master/Run_Model/requirements.txt).

## Via Command Line:
- Clone this repository COVID-Bacterial-Pneumonia-Classifier-Run-Model
- Navigate to the directory : COVID-Bacterial-Pneumonia-Classifier-Run-Model/Run_Model
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

- If you are curious, you may also add your own image to COVID-Bacterial-Pneumonia-Classifier-Run-Model/Run_Model/Sample_Images and predict the classes of those.
