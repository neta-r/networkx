from flask import render_template, url_for, redirect, request

from networkx.web_page import app
from networkx.web_page.templates.forms import Parameters


@app.route("/result")
def result():
    return render_template('resultpage.html')


@app.route("/", methods=['GET', 'POST'])
def home():
    form = Parameters()
    if request.method == 'POST':
        print(form.centrality.data)
        return redirect(url_for('result'))
    else:
        return render_template('homepage.html', form=form)
