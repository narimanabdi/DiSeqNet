import tensorflow as tf
from tensorflow import keras
from .metrics import accuracy

class Senet(keras.Model):

    def compile(
        self, optimizer, metrics, loss_fn,
        ):
        """ Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super().compile(optimizer=optimizer, metrics=metrics)
        self.loss_fn = loss_fn

    def train_step(self, data):
        # Unpack data
        [Xs,Xq],y_true = data
        with tf.GradientTape() as tape:
            #make prediction
            y_pred = self([Xs,Xq],training = True)
            loss =  self.loss_fn(y_true,y_pred)
            #self.loss_tracker.update_state(loss)
        gradients = tape.gradient(loss,self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients,self.trainable_variables))
       
        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y_true, y_pred)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"loss": loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        [Xs,Xq],y_true = data

        # Compute predictions
        y_pred = self([Xs,Xq])

        # Update the metrics.
        self.compiled_metrics.update_state(y_true, y_pred)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        #print( self.compiled_metrics[0])
        return results
    
    def get_config(self):
        config = super(Senet,self).get_config()
        return config