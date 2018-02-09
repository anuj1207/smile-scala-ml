package ml.knoldus.classification

import smile.classification._
import smile.data.AttributeDataset
import smile.validation._
import smile.{data, read}

object RandomForestApp extends App{

  val weather: AttributeDataset = read.arff("src/main/resources/weather.nominal.arff", 4)
  val (trainingInstances,responseVariables) = data.pimpDataset(weather).unzipInt

  private val nTrees = 200
  private val maxNodes = 4
  val rf = randomForest(trainingInstances, responseVariables, weather.attributes(), nTrees, maxNodes)

  val weatherTest = read.arff("src/main/resources/weatherRF.nominal.arff", 4)
  val (testInstances,testResponseVariables) = data.pimpDataset(weatherTest).unzipInt
  println(s"OOB error = ${rf.error}")
  val testedRF = test(trainingInstances, responseVariables, testInstances, testResponseVariables)((_, _) => rf)
}
