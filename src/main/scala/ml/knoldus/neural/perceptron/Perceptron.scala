package ml.knoldus.neural.perceptron

class Perceptron {

  val threshold = 0.5

  def train(inputs: List[Array[Double]], targetOutputs: Array[Double], weights: Array[Double], learningRate: Double, bias: Double, f: (Double, Double) => Double): (Array[Double], Double) = {

    println(s"updated weights are ${weights.toList} and bias is $bias")

    val outputs = inputs.map(input => getOutput(input, weights, bias, f)).toArray

    println(s"outputs are ${outputs.toList}")

    val diffSum = calculateDiffSum(targetOutputs, outputs)

    if(outputs.map(out => {
      f(out, threshold)
    }).zip(targetOutputs).exists{case(exp, gained) => exp != gained}) {
      val (newWeights, newBias) = updateWeightAndBias(weights, inputs, diffSum, learningRate, bias)
      train(inputs, targetOutputs, newWeights, learningRate, newBias, f)
    }
    else
      (weights, bias)
  }

  private def getOutput(input: Array[Double], weightValues: Array[Double], bias: Double, f: (Double, Double) => Double): Double = {
    val summation = input.zip(weightValues).map { case (example, weight) => example * weight }.sum
    f(summation + bias, threshold)
  }

  private def updateWeightAndBias(weights: Array[Double], inputs: List[Array[Double]], diffSum: Double, learningRate: Double, bias: Double): (Array[Double], Double) = {
    inputs match {
      case Nil => (weights, bias)
      case head::tail =>
        val updatedWeights = head.zip(weights).map{case (input, weight) => ( (diffSum * input) * learningRate ) + weight}
        val updatedBias = (diffSum * 1) * learningRate + bias
        updateWeightAndBias(updatedWeights, tail, diffSum, learningRate, updatedBias)
    }
  }

  private def calculateDiffSum(expOut: Array[Double], gainedOut: Array[Double]): Double = {
    expOut.zip(gainedOut).map{case (exp, gained) => exp - gained}.sum
  }
}
