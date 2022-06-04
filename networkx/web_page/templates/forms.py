from flask_wtf import FlaskForm
from wtforms import SubmitField, IntegerField, SelectField


class Parameters(FlaskForm):
    vtx = IntegerField('Number of vertices')
    edges = IntegerField('Number of edges')
    iter = IntegerField('Number of iterations')
    threshold = IntegerField('Threshold')
    centrality = SelectField(u'Centrality', choices=[('cl', 'Closeness'), ('bt', 'Betweeness'), ('dg', 'Degree')])
    type = SelectField(u'Initialization algorithm', choices=[('Comp', 'Complete algorithm'), ('Cyc', 'Cycle algorithm'),
                                                             ('Str', 'Start algorithm'), ('Wh', 'Wheel algorithm')])
    submit = SubmitField('Submit')
