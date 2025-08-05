

import keras
import numpy as np
import json

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
            - classified: Classic classifier network
            - linear-bottleneck: Approximate categories using a linear regression, bottleneck, and category approximator
        """
        self.model_type = model_type
        self.model = None
        self.encode_layer = None
        self.cat_approx_layer = None
        self.validation_data = None
        self.data_generator = GenerateData()
        
    def build_bottleneck_model(self):
        """Build a bottleneck model"""
        input_tensor = keras.Input(shape=(config["num_inputs"],))
        self.encode_layer = keras.layers.Dense(1)  # creates the 1-wide bottleneck
        t = self.encode_layer(input_tensor)
        
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
        return input_tensor, t
    
    def build_classifier_model(self):
        """Build a classic classifier model"""
        input_tensor = keras.Input(shape=(config["num_inputs"],))
        t = input_tensor
        for layer_num in range(6):
            t = keras.layers.Dense(config["num_inputs"], activation="relu")(t)
        t = keras.layers.Dense(config["num_categories"])(t)
        t = keras.layers.Softmax(axis=-1)(t)
        return input_tensor, t

    def build_model(self):
        """Build the keras model"""
        if self.model_type in ["linear-bottleneck"]:
            input_tensor, outputs = self.build_bottleneck_model()
        if self.model_type == "classifier":
            input_tensor, outputs = self.build_classifier_model()
        
        self.model = keras.Model(inputs=input_tensor, outputs=outputs)
        self.model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            optimizer=keras.optimizers.Adam(),
            metrics=["accuracy"]
        )

    def print_weights(self):
        if self.model_type != "linear-bottleneck":
            return
        e = self.encode_layer.get_weights()
        a = self.cat_approx_layer.get_weights()
        data = {
           "encode": {
               "weights": e[0].tolist(),
               "biases": e[1].tolist()
            },
           "cat_appox": {
               "weights": a[0].tolist(),
               "biases": a[1].tolist()
            }
        }
        print(json.dumps(data))

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
            r = self.data_generator.rnd_inputs()
            inputs.append(self.data_generator.normalize_inputs(r))
            targets.append([self.data_generator.input_to_category(r)[0]])
        return np.array(inputs, dtype="float32"), np.array(targets)
        

class ScoringCallback(keras.callbacks.Callback):
    """Print weights/biases after every epoch"""
        
    def __init__(self, score_model):
        self.score_model = score_model;
        
    def on_epoch_end(self, epoch, logs):
        print("")
        print(f"Epoch #{epoch+1} finished. Logs: {logs}")
        self.score_model.print_weights()

