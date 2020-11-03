## Steps to run the code

1. Package Requirements
- pip install numpy

2. Make sure that the `20_newsgroups` folder is inside another directory called `data` which is at the same level where the code file `naive_bayes.py` is
- The path for data used in the code is `data_dir = 'data/20_newsgroups/'` 

3. Run the code with `python naive_bayes.py`
- It runs a 10-fold validation on the data
- Also, handles exceptions when some documents cannot be decoded and ignores them