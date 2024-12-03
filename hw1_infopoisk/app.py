from flask import Flask, request, render_template
from search_tfidf import TfidfSearcher
from searcher_fasttext import FastTextSearcher


app = Flask(__name__, template_folder="templates")

tf_idf = TfidfSearcher(matrix_file_name='tfidf_index_matrix.pickle')
fasttext = FastTextSearcher(fasttext_index_matrix='fasttext_index_matrix.pickle')


@app.route('/')
def initial():
    return render_template("index.html")


@app.route('/search', methods=['GET'])
def search():
    try:  # log exceptions
        if request.args:
            if "n" in request.args:  
                n = request.args["n"]
                if n:
                    n = int(n)
                else:
                    n = 10
            else:  
                n = 10
            metrics = []
            text = request.args["query_text"]  
            if "engine" in request.args:  
                engine = request.args["engine"]
            else:  
                engine = "tf-idf"
            if engine == "tf-idf":
                metrics = tf_idf.search(text, n=n)
            elif engine == "fasttext":
                metrics = fasttext.search(text, n=n)
            if not metrics[0][0]:
                return render_template("index.html", text=text, engine=engine)
            metrics = [item for item in metrics if item[0]]
            if n != len(metrics):
                n = len(metrics)
            return render_template("index.html", text=text, engine=engine, n=n, metrics=metrics)
        else:
            return render_template("index.html")
    except Exception as ex:  
        return render_template("index.html", exception=ex)


if __name__ == "__main__":
    app.run()