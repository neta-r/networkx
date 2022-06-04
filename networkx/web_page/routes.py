from flask import render_template, flash, url_for, redirect, request

from networkx.web_page import app
from networkx.web_page.templates.forms import Parameters


@app.route("/result")
def result():
    return "hello world"


@app.route("/", methods=['GET', 'POST'])
def home():
    form = Parameters()
    if request.method == 'POST':
        return redirect(url_for('result'))
    else:
        return render_template('homepage.html', form=form)
