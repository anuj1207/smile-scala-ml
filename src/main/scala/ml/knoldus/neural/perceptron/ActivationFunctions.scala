package ml.knoldus.neural.perceptron

class ActivationFunctions {

  def stepFunction : (Double, Double)=> Double = {
    (output: Double, threshold: Double) =>  if(output < threshold) 0.0 else 1.0
  }

}
