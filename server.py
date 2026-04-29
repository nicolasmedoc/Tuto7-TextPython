from flask import Flask,request
from flask_cors import CORS
import dataset
import textprocessing
import dimred
import projection
import clustering

app = Flask(__name__)
# allows cross origin to be called from localhost:3000
# not recommended in production
CORS(app)

# insert code for server initialization if needed
dataset, true_k = dataset.get20newsgroups()
x_tfidf, vectorizer = textprocessing.get_tfidf(dataset.data)
x_lsa, lsa = dimred.lsa(x_tfidf)
proj_euclidean_lsa = projection.tsne_euclidean(x_lsa)
proj_cosine_tfidf = projection.tsne_cosine(x_tfidf)
proj_euclidean_tfidf = projection.tsne_euclidean_tfidf(x_tfidf)
clustering_model = clustering.kmeans(4, x_tfidf)
clustering_labels = clustering_model.labels_

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/getProjection")
def get_projection():
    distance = request.args.get("distance");
    proj = None
    if distance=="euclidean_lsa":
        proj = proj_euclidean_lsa
    elif distance=="cosine_tfidf":
        proj = proj_cosine_tfidf
    elif distance == "euclidean_tfidf":
        proj = proj_euclidean_tfidf
    return {"projection": proj.tolist(), "categories": dataset.target.tolist(), "kmeans": clustering_labels.tolist()}