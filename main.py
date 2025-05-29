import os
import pandas as pd
import numpy as np
import re
import psutil
import umap
import time
import warnings
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans,DBSCAN
from TextEmbedding.text_embedding import TextEmbedding
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.metrics import f1_score,silhouette_score
from scipy.optimize import linear_sum_assignment


def read_Vectorized_File(f_Name):
    df = pd.read_csv(f_Name)
    datas = df['vectorized_text'].values
    data_list = []
    for data in datas:
        data_list.append(np.fromstring(data[1:-1], dtype=float, sep=','))
    data_points = np.array(data_list)
    global k
    actual_clusters = df['cluster']
    if (f_Name.__contains__("AG")):
        k = 4
    elif (f_Name.__contains__("Fake")):
        k = 2
    elif (f_Name.__contains__("amazon")):
        k = 5
    elif (f_Name.__contains__("bbc")):
        k = 5
    elif (f_Name.__contains__("dbpedia")):
        k = 14
    elif (f_Name.__contains__("NewsGroup")):
        k = 3
    elif (f_Name.__contains__("Reuters")):
        k = 2
    elif (f_Name.__contains__("Stance")):
        k = 5
    elif (f_Name.__contains__("Illness")):
        k = 4
    return data_points,actual_clusters
def processText(dataFrame, modelname,method,dataSetName):
    match modelname:
        case "BERT":
            modelFullName = 'sentence-transformers/bert-base-nli-mean-tokens'
            max_seq = 128
        case "ROBERTA":
            modelFullName = 'sentence-transformers/nli-roberta-base-v2'
            max_seq = 75
        case "DISTILBERT":
            modelFullName = 'sentence-transformers/distilbert-base-nli-mean-tokens'
            max_seq = 128
        case "ALBERT":
            modelFullName = 'sentence-transformers/paraphrase-albert-small-v2'
            max_seq = 100
        case "MPNET":
            modelFullName = 'sentence-transformers/all-mpnet-base-v2'
            max_seq = 128
        case "DISTILROBERTA":
            modelFullName = 'sentence-transformers/all-distilroberta-v1'
            max_seq = 128
        case "DISTILBERT_v2":
            modelFullName = 'sentence-transformers/multi-qa-distilbert-cos-v1'
            max_seq = 128
        case "T5":
            modelFullName = 'sentence-transformers/sentence-t5-base'
            max_seq = 128
        case "LONGFORMER":
            modelFullName = 'allenai/longformer-base-4096'
            max_seq = 4096
        case "BIGBIRD":
            modelFullName = 'google/bigbird-roberta-base'
            max_seq = 4096

    documents = dataFrame.get('text').tolist()
    actual_clusters = dataFrame.get('cluster')

    processor = TextEmbedding(documents, modelFullName, method,max_seq)
    dataPoints = processor.data
    normalizer = Normalizer()
    normalized_dataPoints = normalizer.fit_transform(dataPoints)
    return normalized_dataPoints,actual_clusters
def read_File(filename) :
    data_frame = pd.read_csv(filename, delimiter=",")
    data_frame["text"] = data_frame.get('text').apply(transformar_tokenizer)
    global k
    if (filename.startswith("AG_News")):
        k = 4
    elif (filename.startswith("Fake")):
        k = 2
    elif (filename.startswith("amazon")):
        k = 5
    elif (filename.startswith("bbc")):
        k = 5
    elif (filename.startswith("dbpedia")):
        k = 14
    elif (filename.startswith("NewsGroup")):
        k = 3
    elif (filename.startswith("Reuters")):
        k = 2
        data_frame['title']=data_frame.get('title').apply(transformar_tokenizer)
        data_frame['text']=data_frame['title']+". "+data_frame['text']
    if (filename.startswith("stance")):
        k = 5
    if (filename.startswith("illness")):
        k = 4
    if (filename.startswith("yahoo")):
        k = 10
        data_frame['question'] = data_frame.get('question').apply(transformar_tokenizer)
        data_frame['text'] = data_frame['question'] + " " + data_frame['text']
    if (filename.startswith("uci")):
        k = 823
    return data_frame
def transformar_tokenizer(text):
    text=remove_URLs(text)
    text = re.sub(r'([^a-zA-Z ?.\'])+', '', text)
    text = re.sub(r'([^a-zA-Z])\1+', r'\1', text)
    text = re.sub('([.?)])', r'\1 ', text)
    return text
def remove_URLs(text):
    from urllib.parse import urlparse
    # global text_data
    text = ' '.join([c for c in text.split() if not urlparse(c).scheme])  # urlparse method solution
    # text_data = re.sub('http[s]?://\S+', '', text_data)  # Regular Expression Solution
    return text
def run_KMeans(data_points, i):
    k_means = KMeans(n_clusters=k, init='random',n_init=10,random_state=i)
    k_means.fit(data_points)
    predicted_clusters = k_means.labels_
    return predicted_clusters
def run_Medoids(data_points,i):
    kmedoids = KMedoids(n_clusters=k, random_state=i, metric="euclidean")
    predicted_clusters = kmedoids.fit_predict(data_points)
    return predicted_clusters
def Umap_Dimention_Reduction(data_points):
    warnings.filterwarnings('ignore')
    seed=42
    reducer = umap.UMAP(n_neighbors=70, n_components=5, metric="euclidean", min_dist=0.9,random_state=seed,transform_seed=seed)
    reduced_data_points = reducer.fit_transform(data_points)
    return reduced_data_points
def Calculate_F1_Score(actual, predicted, n_clusters):  # Maps clusters to actual labels (Bu kod Grok AI 'a yazdırıldı)
    cost_matrix = np.zeros((n_clusters, n_clusters))
    unique_actual = np.unique(actual)  # Örn: [1, 2, 3]
    unique_pred = np.unique(predicted)  # Örn: [1, 2, 3]
    for i in range(n_clusters):
        for j in range(n_clusters):
            cost_matrix[i, j] = -sum((actual == unique_actual[i]) & (predicted == unique_pred[j]))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # Mapping: predicted -> actual
    mapping = {unique_pred[j]: unique_actual[i] for i, j in zip(row_ind, col_ind)}
    mapped_labels = np.array([mapping[label] for label in predicted])
    f1=f1_score(actual, mapped_labels, average='weighted')  # F1 Score with matched clusters
    return f1
def clusterAndlist_Results(data_points, actual_clusters,metric_writer,clustering_method,method_name,dataset):
    metric_array = []
    for i in range(20):
        start_Time = time.time()
        if clustering_method=="K-Means":
            pred_clusters = run_KMeans(data_points, i)
        elif clustering_method=="K-Medoids":
            pred_clusters=run_Medoids(data_points, i)
        Silh_ = silhouette_score(data_points, pred_clusters)
        RS_ = metrics.rand_score(actual_clusters, pred_clusters)  # Rand_score
        F1_ = Calculate_F1_Score(actual_clusters, pred_clusters,k)  # Matching cluster before f1-score calculatation
        ARI_ = metrics.adjusted_rand_score(actual_clusters, pred_clusters)
        NMI_ = metrics.normalized_mutual_info_score(actual_clusters, pred_clusters, average_method='geometric')
        end_Time = time.time()
        Run_Time=end_Time-start_Time
        metric_values = [RS_, ARI_, NMI_, Silh_, F1_, Run_Time]
        metric_array.append(metric_values)
    print("Clustering Method:"+clustering_method+" Dataset:"+dataset+" Method:"+method_name+ " Model: " + model + " Running Time::{}".format(end_Time - start_Time))
    np_metric_array = np.array(metric_array)
    avg = np.mean(np_metric_array, axis=0)
    std = np.std(np_metric_array, axis=0)  # Standart sapmayı hesapla
    np_metric_array = np.vstack((np_metric_array, avg, std))  # Ortalama ve standart sapmayı sona ekler

    df_excel = pd.DataFrame(np_metric_array, columns=['RS', 'ARI', 'NMI', 'SIL', 'F1', 'RunTime'])
    df_excel = df_excel.round(3)  # Virgülden sonra 3 basamak hassasiyet
    df_excel.to_excel(metric_writer, sheet_name=model,index=False)

if __name__ == '__main__':
    SBertModels=["BERT","ROBERTA","DISTILBERT","ALBERT","MPNET", "LONGFORMER","BIGBIRD"]
    # method=> 1:SL, 2:BL, 3:Default, 4:Longformer, 5:BigBird
    methods=[1,2,3,4,5]

    #clustering_method ="K-Means","K-Medoids"
    clustering_method ="K-Means"

    data_setList = [ 'amazon_5000',  'bbc_2250', 'Fake_3000']

    for dataset in data_setList:
        excel_filename = f"{dataset}_metric.xlsx"
        with pd.ExcelWriter(excel_filename, engine='openpyxl', mode='w') as writer:
            for method in methods:
                if(method==1):
                    path = clustering_method+"_"+dataset +"_SL_Result_5.xlsx"
                    methodName = 'SL'
                elif (method == 2):
                    path =  clustering_method+"_"+dataset+"_BL_Result_5.xlsx"
                    methodName = 'BL'
                elif(method==3):
                    path = clustering_method+"_"+dataset+"_Default_Result_5.xlsx"
                    methodName = 'Default'
                elif (method == 4):
                    path =  clustering_method+"_"+dataset+"_LongFormer_Result_5.xlsx"
                    methodName = 'LongFormer'
                elif (method == 5):
                    path = clustering_method+"_"+dataset+"_BigBird_Result_5.xlsx"
                    methodName = 'BigBird'

                dataFrame = read_File(dataset+".csv")

                metric_list=[]
                with pd.ExcelWriter(path, engine='openpyxl', mode='w') as metric_writer:
                    for model in SBertModels:
                        if (model=="LONGFORMER" and method!=4) :
                            continue
                        if(model!="LONGFORMER" and method==4):
                            continue
                        if model=="BIGBIRD" and method!=5:
                            continue
                        if model!="BIGBIRD" and method==5:
                            continue
                        process = psutil.Process(os.getpid()) #To calculate Memory Usage
                        start_time = time.time() #To calculate runtime

                        data_points,actual_clusters=processText(dataFrame,model,method,dataset)
                        data_points = Umap_Dimention_Reduction(data_points)
                        clusterAndlist_Results(data_points, actual_clusters,metric_writer,clustering_method,methodName,dataset)

                        end_time=time.time() #To calculate runtime
                        run_time=end_time-start_time#To calculate runtime

                        memory_usage = process.memory_info().rss / 1024 ** 2 #To calculate Memory Usage in MB

                        word_counts = dataFrame["text"].apply(lambda x: len(str(x).split())) #To calculate throughput as texts/second
                        throughput = word_counts.sum() / run_time  # texts/second

                        print(f"Dataset:{dataset}, Method:{method}, Model:{model} - Runtime: {run_time:.2f}s, Memory: {memory_usage:.2f}MB, Throughput: {throughput:.2f} texts/s")
                        metric_values = [model,run_time, memory_usage, throughput]
                        metric_list.append(metric_values)
                        # Convert metric_list to a DataFrame
                    df = pd.DataFrame(metric_list, columns=['Model','Run-Time', 'Memory-Usage', 'Throughput'])
                    df = df.round(3)
                    df.to_excel(writer, sheet_name=methodName, index=False)














