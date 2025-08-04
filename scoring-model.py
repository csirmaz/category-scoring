


import keras
import numpy as np
import json


config = {
    "num_inputs": 8,
    "num_categories": 4,
    "steps_per_epoch": 10000,
    "validation_steps": 50,
    "epochs": 100,
}



class ScoringModel:
    """Define and train a model"""
    
    def __init__(self):
        # TODO types: regular NN, NN+bottlenect, linear+bottlenect
        self.model = None
        self.encode_layer = None
        self.cat_approx_layer = None

    def build_model(self):
        """Build the keras model"""
        input_tensor = keras.Input(shape=(config["num_inputs"],))
        self.encode_layer = keras.layers.Dense(1)  # creates the 1-wide bottleneck
        t = self.encode_layer(input_tensor)
        
        self.cat_approx_layer = keras.layers.Dense(config["num_categories"], kernel_constraint="NonNeg")
        t = cat_approx_layer(t)

        # TODO down-sum

        t = keras.layers.Softmax(axis=-1)(t)
        
        self.model = keras.Model(inputs=input_tensor, outputs=t)
        self.model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            optimizer=keras.optimizers.Adam(),
            metrics=["accuracy"]
        )

    def print_weights(self):
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
        print(json.dumps(data, indent=4))

    def fit(self):
        self.model.fit(
            x=self.training_data(is_training=True),
            validation_data=self.training_data(is_training=False),
            steps_per_epoch=config["steps_per_epoch"],
            validation_steps=config["validation_steps"],
            epochs=config["epochs"],
            callbacks=[ScoringCallback(self)]            
        )


class ScoringCallback(keras.callbacks.Callback):
    """Print weights/biases after every epoch"""
        
    def __init__(self, score_model):
        self.score_model = score_model;
        
    def on_epoch_end(self, epoch, logs):
        print("")
        print(f"Epoch #{epoch+1} finished. Logs: {logs}")
        self.score_model.print_weights()




