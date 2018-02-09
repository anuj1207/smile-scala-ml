package ml.knoldus.neural

import smile.classification.{NeuralNetwork, mlp}
import smile.validation._
import smile.{data, read}

object NeuralNet extends App {
  val weather = read.arff("src/main/resources/weatherRF.nominal.arff", 4)
  /*returns object of attributeDataSet*/
  val (trainingInstances, responseVariables) = data.pimpDataset(weather).unzipInt

  private val epochs = 30 //the number of epochs of stochastic learning.
  private val eta = 0.1 //the learning rate.
  private val alpha = 0.2 //the momentum factor.
  private val lambda = 0.1 //the weight decay for regularization.

  val nNetwork = mlp(trainingInstances, responseVariables, Array(4, 2),
    NeuralNetwork.ErrorFunction.LEAST_MEAN_SQUARES,
    NeuralNetwork.ActivationFunction.LINEAR, epochs, eta, alpha, lambda)

  val weatherTest = read.arff("src/main/resources/weatherTest.nominal.arff", 4)
  val (testInstances, testResponseVariables) = data.pimpDataset(weatherTest).unzipInt
  //  println(s"OOB error = ${nNetwork.error}")

  val decisions = testInstances.map {
    nNetwork.predict(_) match {
      case 0 => "play"
      case 1 => "not playable weather"
    }
  }.toList

  println(decisions)

  val error = testInstances.zip(testResponseVariables).count {
    case (testInstance, response) => nNetwork.predict(testInstance) != response
  }

  println("Number of errors in test data is " + error)

  val testedRF = test(trainingInstances, responseVariables, testInstances, testResponseVariables)((_, _) => nNetwork)
}
