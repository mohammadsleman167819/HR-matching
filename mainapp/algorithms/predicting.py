import joblib
import gensim

def predict(text):
        
        from pathlib import Path
        BASE_DIR = Path(__file__).resolve().parent
        kmeans_path = Path(BASE_DIR, 'Kmeans_model.joblib')  
        #if not kmeans_path.is_file():
         #   return None
        
        word2vecmodel_path = Path(BASE_DIR, 'word2vecmodel.joblib')  
       # if not word2vecmodel_path.is_file():
        #    return None
        
        model = joblib.load(kmeans_path)
        word2vecmodel = joblib.load(word2vecmodel_path)
        try:
                vector = gensim.utils.simple_preprocess(text)
                w2v = word2vecmodel.wv.get_mean_vector(vector)
                vector_2d = w2v.reshape(1,-1)
                label = model.predict(vector_2d)
                return label
        except:
                return -1
