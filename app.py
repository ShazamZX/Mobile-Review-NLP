
from flask import Flask, render_template, request
import num2word
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from num2words import num2words
import pickle
def clean_text(st):
    def convert_func(matchobj):
      m =  matchobj.group(0)
      return num2words(m)      
    res = re.sub('[0-9]+', convert_func, st)
    res = re.sub('[^a-zA-Z]',' ',res)
    return res.strip()
def lemma(data):
    lt = WordNetLemmatizer()
    corpus = ''
    datax = data.split()
    impwords = ['no','nor','not','only','few','more','most','all']
    for i in datax:
        ch = i.lower()
        words = ch.split()
        words = [lt.lemmatize(word) for word in words if (word not in stopwords.words('english')) and (word not in impwords)]
        words = ''.join(words)
        if (words != ''):
            corpus = corpus + ' ' + words
    return (corpus.strip())
classifier = pickle.load(open('MobileReviewModel.pkl', 'rb'))
vect = pickle.load(open('vectorizer.pkl', 'rb'))


app = Flask(__name__)

@app.route('/', methods= ['GET'])
def Home():
    return render_template('index.html')
@app.route('/predict', methods = ['POST'])
def predict():
    if request.method == 'POST':
        str = request.form['review']
        str = clean_text(str)
        x = lemma(str)
        corpus = [x]
        predvect = vect.transform(corpus).toarray()
        predval = classifier.predict(predvect)
        output = int(predval[0])
        if output == 0:
            return render_template('predict.html', imgdata = '../static/negative.jpg', textout="OOPS! CUSTOMER SEEMS DISAPPOINTED WITH THE PHONE!")
        else:
            return render_template('predict.html', imgdata = '../static/positive.jpg',textout="WOW! CUSTOMER SEEMS TO LIKE THE PHONE!")
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)