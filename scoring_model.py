

import keras
import numpy as np
import json
import math

from generate_data import GenerateData


config = {
    "num_inputs": 8,
    "num_categories": 3,
    "steps_per_epoch": 10000,
    "validation_steps": 500,
    "epochs": 100,
    "batch_size": 64,
}



class ScoringModel:
    """Define and train a model"""
    
    def __init__(self, model_type):
        """Build and train a model. Available types:
            - classifier: Classic classifier network
            - predict-score: Predict pre-set scores
            - linear-bottleneck: Approximate categories using a linear regression, bottleneck, and category approximator
        """
        self.model_type = model_type
        self.model = None
        self.encode_layer = None
        self.cat_approx_layer = None
        self.data_generator = GenerateData()

        
    def build_linear_bottleneck_model(self):
        """Build the final, linear bottleneck model"""
        input_tensor = keras.Input(shape=(config["num_inputs"],))
        self.encode_layer = keras.layers.Dense(1)  # encoder - creates the 1-wide bottleneck
        t = self.encode_layer(input_tensor)
        
        # decoder
        self.cat_approx_layer = keras.layers.Dense(config["num_categories"], kernel_constraint="NonNeg")
        t = self.cat_approx_layer(t)
        
        # Propagate-sum: turn [a, b, c] into [a, a+b, a+b+c]
        values = [t[..., i:i+1] for i in range(config["num_categories"])]
        values2 = []
        for i in range(config["num_categories"]):
            track = [0]
            s = values[0]
            for j in range(1, i+1):
                track.append(j)
                s += values[j]
            print(f"Propagate sum: #{i} becomes {track}")
            values2.append(s)
        t = keras.layers.Concatenate()(values2)

        t = keras.layers.Softmax(axis=-1)(t)
        
        self.model = keras.Model(inputs=input_tensor, outputs=t)
        self.model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            optimizer=keras.optimizers.Adam(),
            metrics=["accuracy"]
        )

    
    def build_classifier_model(self):
        """Build a classic classifier model"""
        input_tensor = keras.Input(shape=(config["num_inputs"],))
        t = input_tensor
        for layer_num in range(6):
            t = keras.layers.Dense(config["num_inputs"], activation="relu")(t)
        t = keras.layers.Dense(config["num_categories"])(t)
        t = keras.layers.Softmax(axis=-1)(t)

        self.model = keras.Model(inputs=input_tensor, outputs=t)
        self.model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            optimizer=keras.optimizers.Adam(),
            metrics=["accuracy"]
        )


    def build_predict_score_model(self):
        """Build a score predictor model with a single output"""
        input_tensor = keras.Input(shape=(config["num_inputs"],))
        t = input_tensor
        for layer_num in range(2):
            t = keras.layers.Dense(config["num_inputs"], activation="relu")(t)
        t = keras.layers.Dense(1)(t)

        self.model = keras.Model(inputs=input_tensor, outputs=t)
        self.model.compile(
            loss=keras.losses.MeanSquaredError(),
            optimizer=keras.optimizers.Adam(),
            metrics=[keras.metrics.MeanAbsoluteError()]
        )
        

    def build_model(self):
        """Build the keras model"""
        if self.model_type == "classifier":
            self.build_classifier_model()
        if self.model_type == "predict-score":
            self.build_predict_score_model()
        if self.model_type == "linear-bottleneck":
            self.build_linear_bottleneck_model()
        self.model.summary()


    def fit(self):
        self.model.fit(
            x=self.training_data(is_training=True),
            validation_data=self.training_data(is_training=False),
            steps_per_epoch=config["steps_per_epoch"],
            validation_steps=config["validation_steps"],
            epochs=config["epochs"],
            callbacks=[ScoringCallback(self)]            
        )

        
    def training_data(self, is_training: bool):
        """Generator for training and validation data"""
        while True:
            i, t = self.get_data_batch()
            yield i, t

                
    def get_data_batch(self):
        """Return a batch of data from the toy example"""
        inputs = []
        targets = []
        for i in range(config["batch_size"]):
            raw_inputs = self.data_generator.rnd_inputs()
            inputs.append(self.data_generator.normalize_inputs(raw_inputs))
            targets.append([self.data_generator.input_to_category(raw_inputs)[0]])
        if self.model_type == "predict-score":
            return np.array(inputs, dtype="float32"), np.array(targets, dtype="float32")
        return np.array(inputs, dtype="float32"), np.array(targets)

        
    def eval_bottleneck_model(self):
        print("----- Evaluating the bottleneck model -----")
        # Get the weights and biases
        e = self.encode_layer.get_weights()
        a = self.cat_approx_layer.get_weights()
        data = {
           "encode": {
               "weights": e[0].tolist(),
               "biases": e[1].tolist()
            },
           "cat_approx": {
               "weights": a[0].tolist(),
               "biases": a[1].tolist()
            }
        }
        # print(json.dumps(data))
        
        # Gather the linear model
        weights = [w[0] for w in data["encode"]["weights"]]
        raw_thresholds = self.extract_thresholds(data)
        thresholds = [t - data['encode']['biases'][0] for t in raw_thresholds]
        
        # Print parameters
        input_labels = self.data_generator.input_labels()
        for ix, w in enumerate(weights):
            print(f"{input_labels[ix]} weight: {w}")
            
        for t in thresholds:
            print(f"Threshold: {t}")
            
        # Try the linear model
        self.try_linear_model(weights, thresholds)
        
        print("----- end -----")


    def extract_thresholds(self, data):
        """Extract thresholds from the cat_approx layer"""
        ca = data["cat_approx"]
        # Don't forget that we apply a cascading sum
        # Extracting the actual weights and biases
        length = len(ca["weights"][0])
        weights = [0 for i in range(length)]
        biases = [0 for i in range(length)]
        for i in range(length):
            for j in range(i, length):
                weights[j] += ca["weights"][0][i]
                biases[j] += ca["biases"][i]
                
        # Now we know that the weights are in increasing order, but let's sanity check
        for i in range(length-1):
            assert weights[i] <= weights[i+1]
        
        # Solve the equations
        thresholds = []
        for i in range(length-1):
            db = biases[i+1] - biases[i]
            dw = weights[i] - weights[i+1]
            thresholds.append("inf" if dw == 0 else db / dw)
        # print(f"Raw thresholds between categories: {thresholds}")
        return thresholds
    
    
    def try_linear_model(self, weights, thresholds):
        """Try the linear model learnt by the linear_bottleneck model"""
        samples = 10
        for i in range(samples):
            raw_inputs = self.data_generator.rnd_inputs()
            normalized_inputs = self.data_generator.normalize_inputs(raw_inputs)
            target = self.data_generator.input_to_category(raw_inputs)[0]
            
            # Calculate linear combination
            score = 0
            for j, v in enumerate(normalized_inputs):
                score += weights[j] * v
                
            #print(f"Raw inputs: {raw_inputs}")
            #print(f"Normalized: {normalized_inputs}")
            #print(f"Score: {score}")
            
            # Use thresholds to predict category
            if thresholds[0] > thresholds[1]:
                # The middle category is never predicted
                category = "n/a"
            elif score < thresholds[0]:
                category = 0
            elif score < thresholds[1]:
                category = 1
            else:
                category = 2
            
            print(f"Sample #{i}: target={target} score={score:+.2f} predicted={category} {'ok' if category==target else 'x'}")
        
        
    def eval_score_predictor(self):
        """Evaluate the score predictor model"""
        print("----- Evaluating the score predictor model -----")
        y = self.model.predict_on_batch(self.get_data_batch()[0])
        # visualize the values
        # -.5 to 2.5
        histogram = [0 for i in range(60)]
        for v in y:
            v = v[0]
            if v < -.5 or v >= 2.5:
                continue
            v = int((v + .5) * 20.)
            histogram[v] += 1
        vertical_step = max(histogram) / 10.
        for row in range(10):
            for v in histogram:
                print("#" if v > (10 - row - 1) * vertical_step else ".", end="")
            print("")
        print("----- end -----")
        

class ScoringCallback(keras.callbacks.Callback):
    """Print weights/biases after every epoch"""
        
    def __init__(self, score_model):
        self.score_model = score_model;
        
    def on_epoch_end(self, epoch, logs):
        print("")
        print(f"Epoch #{epoch+1} finished. Logs: {logs}")
        if self.score_model.model_type == "linear-bottleneck":
            self.score_model.eval_bottleneck_model()        
        if self.score_model.model_type == "predict-score":
            self.score_model.eval_score_predictor()

