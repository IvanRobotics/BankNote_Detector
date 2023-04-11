# 1. import libraries
from fastapi import FastAPI
from app.model.model import __version__ as model_version
from app.model.model import classifiers
from app.model.model import BankNote


# 2. create the app object
app = FastAPI()


# 3. Index and route
@app.get('/')
def index():
    return {'health_check': 'OK', "model_version": model_version }

# 4. Prediction
@app.post('/predict')
def predict_banknote(data:BankNote, choice: int = 1):

    classifier_names = []
    for index, clf in enumerate(classifiers):
        classifier_names.append((index,clf.__class__.__name__))

    select_classifier = classifiers[choice-1]

    data = data.dict()
    variance = data['variance']
    skewness = data['skewness']
    curtosis = data['curtosis']
    entropy  = data['entropy']
    prediction = select_classifier.predict([[variance,skewness,curtosis,entropy]])
    
    if prediction[0] > 0.5:
        return {'result': 'Fake Note', 'available_classifiers': classifier_names}

    
    else:
        return {'result': 'It is a bank note!', 'available_classifiers': classifier_names}

     
# Run the API with uvicorn
# if __name__ == '__main__':
#     uvicorn.run(app, host ='localhost', port = 8080)