

Install dependencies
```
pip install Flask torch torchvision nltk
```
Install nltk package
```
$ python
>>> import nltk
>>> nltk.download('punkt')
```
Modify `intents.json` 

Run
```
$  python train.py
```
This will dump data.pth file. And then run
the following command to test it in the console.
```
$  python chat.py
```

