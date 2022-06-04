from flask import render_template, flash, url_for, redirect

from networkx.web_page import app
from networkx.web_page.templates.forms import Parameters


@app.route("/result")
def result():
    return "hello world"


@app.rout("/", methods=['GET', 'POST'])
def home():
    form = Parameters()
    if form.validation_on_submit():
        flash(f'Information received successfully!', 'success')
        return redirect(url_for(home.__home__))
    return render_template('homepage.html', title='Param', form=form)
