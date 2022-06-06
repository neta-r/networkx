from flask_wtf import FlaskForm
from wtforms import SubmitField, IntegerField, SelectField
from wtforms.validators import DataRequired


class Parameters(FlaskForm):
    vtx = IntegerField('Number of vertices', validators=[DataRequired()])
    edges = IntegerField('Number of edges', validators=[DataRequired()])
    gravity = IntegerField('Gravity', default=6)
    iter = IntegerField('Number of iterations', default=50)
    centrality = SelectField(u'Centrality', choices=[('cl', 'Closeness'), ('bt', 'Betweeness'), ('dg', 'Degree')])
    type = SelectField(u'Initialization algorithm', choices=[('Comp', 'Complete algorithm'), ('Cyc', 'Cycle algorithm'),
                                                             ('Str', 'Start algorithm'), ('Wh', 'Wheel algorithm')])
    submit = SubmitField('Run!')
