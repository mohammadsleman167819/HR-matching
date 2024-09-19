import numpy as np
from ..models import Job_Post,cluster_records,Course,Employee,globals  
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import silhouette_score
import joblib
import gensim
from .predicting import predict
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os


def gensim_fun(word2vec_window_size,
               word2vec_word_min_count_percentage,
               word2vec_vector_size,
               text_data,dump=0):

    #Calculate the minimum document frequency for a word to be considered based on the percentage passes
    word2vec_word_min_count = int(len(text_data) * word2vec_word_min_count_percentage)

    #initilize the model
    word2vecmodel = gensim.models.Word2Vec(
          window = word2vec_window_size,
          min_count = word2vec_word_min_count,
          vector_size = word2vec_vector_size)

    #vectorize docs using gensim preprocessing
    corpus_iterable =[]
    for text in text_data:
        vector = gensim.utils.simple_preprocess(text)
        corpus_iterable.append(vector)

    #build vocabulary and train word2vec model
    word2vecmodel.build_vocab(corpus_iterable)
    word2vecmodel.train(corpus_iterable,
                        total_examples=word2vecmodel.corpus_count,
                        epochs = word2vecmodel.epochs)
    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parent
    word2vecmodel_path = Path(BASE_DIR, 'word2vecmodel.joblib')  

    if(dump):   
        joblib.dump(word2vecmodel, word2vecmodel_path)
    #replace each doc with a vector calculated as mean of all words vectors in the doc
    vectors=[]
    for text in corpus_iterable:
        vectors.append(word2vecmodel.wv.get_mean_vector(text))

    #change the diminsions of the vectors array to be suitable for training functions
    vectors_2d = np.stack(vectors)

    return vectors_2d




def kmeans_fun(n_clusters,max_iter,n_init,vectors,dump=0):

    model = KMeans(n_clusters = n_clusters, init='k-means++', max_iter=max_iter,n_init=n_init)
    labels = model.fit_predict(vectors)


    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parent
    kmeans_path = Path(BASE_DIR, 'Kmeans_model.joblib')  
    if(dump):
        joblib.dump(model, kmeans_path)  
    

    cluster_labels = model.labels_
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)

    # Print the number of data points in each cluster
    print("Number of data points in each cluster:")
    for label, count in zip(unique_labels, counts):
      print(f"Cluster {label+1}: {count}")
    print("===========")
    return {'labels':labels,'inertia':model.inertia_}



def sil_fun(vectors,labels):
    sil_score = silhouette_score(vectors, labels)
    sil_score = round(sil_score,3)
    return sil_score

def ch_fun(vectors,labels):
    ch_score = calinski_harabasz_score(vectors, labels)
    ch_score = round(ch_score,3)
    return ch_score


from collections import defaultdict

def delete_old_plots():
    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parent.parent
    static_folder = os.path.join(BASE_DIR, 'static/Images')  
    filename = "sil_score.png" 
    filepath = os.path.join(static_folder, filename)
    print("deleting")
    if os.path.exists(filepath):
        os.remove(filepath)
        print("1")
    filename = "elbow_method.png" 
    filepath = os.path.join(static_folder, filename)

    if os.path.exists(filepath):
        # Delete the existing file
        os.remove(filepath)
        print("2")
    filename = "ch_score.png" 
    filepath = os.path.join(static_folder, filename)
    
    
    if os.path.exists(filepath):
        # Delete the existing file
        os.remove(filepath)
        print("3")
        

def plot_arrays():
    
    first_row = globals.objects.first()
    results = list(cluster_records.objects.filter(applied=False).order_by('-id')[:first_row.number_of_rows])
    results = reversed(results)

    data_dict = defaultdict(list)

    
    for record in results:
        for key, value in record.__dict__.items():
            data_dict[key].append(value)

    array1 = data_dict["silhouette_score"]
    array3 = data_dict["calinski_harabasz_score"]
    n_clusters = data_dict["number_of_clusters"]
    array2 = data_dict["inertia"]

    
    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parent.parent
    static_folder = os.path.join(BASE_DIR, 'static/Images')  

    plt.plot(n_clusters, array1, label="Silhouette Score")
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score")
    plt.xticks(n_clusters)
    plt.vlines(n_clusters, ymin=plt.gca().get_ylim()[0], ymax=plt.gca().get_ylim()[1], colors='gray', linestyles='--', alpha=0.5)

    filename = "sil_score.png" 
    filepath = os.path.join(static_folder, filename)
    
    if os.path.exists(filepath):
        os.remove(filepath)
    
    plt.savefig(filepath, format='png')  
    plt.clf()



    plt.plot(n_clusters, array2, label="Elbow Method")
    plt.xlabel("Number of clusters")
    plt.ylabel("inertia")
    plt.title("Elbow Method")
    plt.xticks(n_clusters)
    plt.vlines(n_clusters, ymin=plt.gca().get_ylim()[0], ymax=plt.gca().get_ylim()[1], colors='gray', linestyles='--', alpha=0.5)
    
    
    filename = "elbow_method.png" 
    filepath = os.path.join(static_folder, filename)
    
    
    if os.path.exists(filepath):
        os.remove(filepath)
    
    plt.savefig(filepath, format='png')  
    plt.clf()
    plt.plot(n_clusters, array3, label="Calinski Harabasz Score")
    plt.xlabel("Number of clusters")
    plt.ylabel("Calinski Harabasz Score")
    plt.title("Calinski Harabasz Score")
    plt.xticks(n_clusters)
    plt.vlines(n_clusters, ymin=plt.gca().get_ylim()[0], ymax=plt.gca().get_ylim()[1], colors='gray', linestyles='--', alpha=0.5)
   

    
    filename = "ch_score.png" 
    filepath = os.path.join(static_folder, filename)
    
    
    if os.path.exists(filepath):
        os.remove(filepath)
    
    plt.savefig(filepath, format='png')  
    plt.clf()
    plt.close()





def gensim_kmeans_fun(text_data,
                      word2vec_window_size,
                      word2vec_word_min_count_percentage,
                      word2vec_vector_size,
                      n_clusters,
                      max_iter,
                      n_init,
                      dump=0):

    vectors = gensim_fun( word2vec_window_size,word2vec_word_min_count_percentage,
                          word2vec_vector_size,text_data,dump)
    kmeans_results = kmeans_fun(n_clusters,max_iter,n_init,vectors,dump)
    return {
              'labels' :  kmeans_results['labels'],
              'ch_score' : ch_fun(vectors,kmeans_results['labels']),
              'sil_score' : sil_fun(vectors,kmeans_results['labels']),
              'inertia' : kmeans_results['inertia'],
              'n_clusters' : n_clusters,
              'word2vec_word_min_count_percentage':word2vec_word_min_count_percentage,
           }






def train_model(start_date,
                number_of_clusters,
                word2vec_vector_size,
                word2vec_window_size,
                word2vec_word_min_count_percentage):

    first_row = globals.objects.first()
    first_row.training_thread_running = True
    first_row.save()
    
    job_posts = Job_Post.objects.filter(added_date__gte=start_date)
    number_of_records = job_posts.count()
    if word2vec_word_min_count_percentage < 0.01:
        word2vec_word_min_count_percentage = 0.35    

    text_data = list(job_posts.values_list('clusterable_text', flat=True))
   
    result = gensim_kmeans_fun(text_data=text_data,
                      word2vec_window_size=word2vec_window_size,
                      word2vec_word_min_count_percentage=word2vec_word_min_count_percentage,
                      word2vec_vector_size=word2vec_vector_size,
                      n_clusters=number_of_clusters,
                      max_iter=5000,
                      n_init=10,
                      dump=1)
        
    
    for i in range(len(job_posts)):
        job_posts[i].cluster=result['labels'][i]
        job_posts[i].save()

    job_posts = Job_Post.objects.filter(added_date__lt=start_date)
    for i in range(len(job_posts)):
        job_posts[i].cluster = predict(job_posts[i].clusterable_text)
        job_posts[i].save()

    courses = Course.objects.all()
    for course in courses:
        course.cluster = predict(course.clusterable_text)
        course.save()

    employees = Employee.objects.all()
    for emp in employees:
        emp.cluster = predict(emp.clusterable_text)
        emp.save()
    
    cluster_records.objects.create(calinski_harabasz_score=result['ch_score'], 
                            silhouette_score=result['sil_score'],
                            number_of_clusters=number_of_clusters,
                            total_records=number_of_records,
                            word2vec_vector_size=word2vec_vector_size,
                            word2vec_window_size=word2vec_window_size,
                            word2vec_word_min_count_percentage=word2vec_word_min_count_percentage,
                            from_date = start_date,
                            applied=True,
                            inertia = result['inertia'])

    first_row.training_thread_running = False
    first_row.save()
    return




def test_number_of_clusters_gensim_kmeans(text_data,
                                          start_number, 
                                          end_number, 
                                          step,
                                          word2vec_word_min_count_percentage,
                                          word2vec_vector_size,
                                          word2vec_window_size
                                          ):

    ch_scores = []
    sil_scores = []
    inertias=[]
    results = []
    n_clusters_arr=[]
    
    if word2vec_word_min_count_percentage<0.01:
        word2vec_word_min_count_percentage = 0.35    

    for n_clusters in range(start_number, end_number + 1, step):
        result = gensim_kmeans_fun(text_data=text_data,
                      word2vec_window_size=word2vec_window_size,
                      word2vec_word_min_count_percentage=word2vec_word_min_count_percentage,
                      word2vec_vector_size=word2vec_vector_size,
                      n_clusters=n_clusters,
                      max_iter=5000,
                      n_init=10)
        results.append(result)
        ch_scores.append(result['ch_score'])
        sil_scores.append(result['sil_score'])
        inertias.append(result['inertia'])
        n_clusters_arr.append(n_clusters)
    
    return {'results':results}
    
    
def test_n_clusters(start_clusters, 
                    end_clusters, 
                    step, 
                    start_date,
                    word2vec_vector_size,
                    word2vec_window_size,
                    word2vec_word_min_count_percentage):
 
    first_row = globals.objects.first()
    first_row.testing_thread_running = True
    first_row.save()
    job_posts = Job_Post.objects.filter(added_date__gte=start_date)
    text_data = list(job_posts.values_list('clusterable_text', flat=True))
        
    ret = test_number_of_clusters_gensim_kmeans(text_data=text_data,
                                          word2vec_vector_size = word2vec_vector_size,
                                          word2vec_window_size = word2vec_window_size,  
                                          start_number=start_clusters, 
                                          end_number=end_clusters, 
                                          step=step,
                                          word2vec_word_min_count_percentage=word2vec_word_min_count_percentage)

    results=ret['results']
    for record in results:
        cluster_records.objects.create(
                            calinski_harabasz_score = record['ch_score'], 
                            silhouette_score = record['sil_score'],
                            number_of_clusters = record['n_clusters'],
                            total_records = len(text_data),
                            word2vec_vector_size = word2vec_vector_size,
                            word2vec_window_size = word2vec_window_size,
                            word2vec_word_min_count_percentage = record['word2vec_word_min_count_percentage'],
                            from_date = start_date,
                            applied = False,
                            inertia = record['inertia']
        )
    
    first_row.number_of_rows = len(results)
    first_row.testing_thread_running=False
    first_row.save()
    return 
  
