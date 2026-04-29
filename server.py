from flask import Flask,request
from flask_cors import CORS
import dataset
import textprocessing
import dimred
import projection

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
    return {"projection": proj.tolist(), "categories": dataset.target.tolist()}