package ml.knoldus.neural.perceptron

object PerceptronAND {

  def main(args: Array[String]): Unit = {

    val perceptron = new Perceptron
    val activationFunctions = new ActivationFunctions

    //training dataSet
    val trainingDataSet: List[Array[Double]] = List(Array(1,0), Array(0,1), Array(0, 0), Array(1, 1))
    //outputs that corresponds to AND gate
    val outputs: Array[Double] = Array(0, 0, 0, 1)
    //random weights
    val weights = Array(100.0, 100.0)
    //learning rate
    val learningRate = 0.01

    val bias = 100.0

    val (finalWeights, finalBias) = perceptron.train(trainingDataSet, outputs, weights, learningRate, bias, activationFunctions.stepFunction)

    println("Updated Weights are:")
    finalWeights.foreach(println)
    println(s"and updated bias is : $finalBias")
  }

}
