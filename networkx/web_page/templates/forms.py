from flask_wtf import FlaskForm
from wtforms import SubmitField, IntegerField, SelectField
from wtforms.validators import DataRequired


class Parameters(FlaskForm):
    vtx = IntegerField('Number of vertices', validators=[DataRequired()])
    edges = IntegerField('Number of edges', validators=[DataRequired()])
    iter = IntegerField('Number of iterations', default=50)
    threshold = IntegerField('Threshold', default=70e-4)
    # centrality = SelectField(u'Centrality', choices=[('cl', 'Closeness'), ('bt', 'Betweeness'), ('dg', 'Degree')])
    # type = SelectField(u'Initialization algorithm', choices=[('Comp', 'Complete algorithm'), ('Cyc', 'Cycle algorithm'),
                                                           #  ('Str', 'Start algorithm'), ('Wh', 'Wheel algorithm')])
    submit = SubmitField('Run!')
