from flask import render_template, flash, url_for, redirect

from networkx.web_page import app
from networkx.web_page.templates.forms import Parameters


@app.route("/result")
def result():
    return "hello world"


@app.route("/", methods=['GET', 'POST'])
def home():
    form = Parameters()
    if not form.validate_on_submit():
        return render_template('homepage.html', form=form)
    else:
        flash(f'Welcome, !', 'success')
        return redirect(url_for(result.__name__))
