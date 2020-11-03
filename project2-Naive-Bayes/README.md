## Steps to run the code

1. Package Requirements
- python >= 3.6
- numpy==1.19.1
- pip install numpy

2. Make sure that the `20_newsgroups` folder is inside another directory called `data` which is at the same level where the code file `naive_bayes.py` is
- The path for data used in the code is `data_dir = 'data/20_newsgroups/'` 

3. Run the code with `python naive_bayes.py`
- It runs a 10-fold validation on the data
- Also, handles exceptions when some documents cannot be decoded and ignores them

4. Reports and supporting documents
- `naive_bayes.py` is the actual python script that you can run to train the model
- `code.pdf` contains the jupyter notebook code that I tested on my local machine
- `report.pdf` is the project report
- `stopwords.py` is a list of optimal stop words that I found to work best for my model
- `naive_bayes.ipynb` is the actual jupyter notebook file
