
import argparse

parser = argparse.ArgumentParser(
    prog='train.py',
    usage='%(prog)s --linear-bottleneck|--nonlinear-bottleneck|--classifier',
    description='Train one of three types of models to illustrate learning a score based on categories'
)
parser.add_argument('--linear-bottleneck', action='store_true', help='Train and interpret a linear model with a bottleneck and category approximator head')
parser.add_argument('--predict-score', action='store_true', help='Train a model attempting to predict the score')
parser.add_argument('--classifier', action='store_true', help='Train a classic classifier')
args = parser.parse_args()

if args.linear_bottleneck:
    model_type = "linear-bottleneck"
elif args.predict_score:
    model_type = "predict-score"
elif args.classifier:
    model_type = "classifier"
else:
    parser.error("Unknown model type")
    exit(1)

from scoring_model import ScoringModel

model = ScoringModel(model_type)
model.build_model()
model.fit()
