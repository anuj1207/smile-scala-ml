package ml.knoldus.neural.perceptron

object PerceptronNOT {
  def main(args: Array[String]): Unit = {

    val perceptron = new Perceptron
    val activationFunctions = new ActivationFunctions

    //training dataSet
    val trainingDataSet: List[Array[Double]] = List(Array(1), Array(0))
    //outputs that corresponds to NOT gate
    val outputs: Array[Double] = Array(1, 0)
    //random weights
    val weights = Array(1.9)
    //learning rate
    val learningRate = 0.01

    val bias = 1.9

    val (finalWeights, finalBias) = perceptron.train(trainingDataSet, outputs, weights, learningRate, bias, activationFunctions.stepFunction)

    println("Updated Weights are:")
    finalWeights.foreach(println)
    println(s"and updated bias is : $finalBias")
  }
}
