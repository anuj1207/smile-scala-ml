package ml.knoldus

import java.io.FileInputStream

import smile.classification.DecisionTree

import smile.data.parser.ArffParser
import smile.classification.cart

object DecisionTreeApp extends App{

  val arffParser = new ArffParser()
  arffParser.setResponseIndex(4)
  val weather = arffParser.parse(new FileInputStream("src/main/resources/weather.nominal.arff"))
  val x = weather.toArray(Array(new Array[Double](weather.size())))
  val y = weather.toArray(new Array[Int](weather.size()))

  val dTree = cart(x, y, 200, weather.attributes(), DecisionTree.SplitRule.ENTROPY)

  val tree = dTree.dot()
  println(tree)

  val weatherTest = arffParser.parse(new FileInputStream("src/main/resources/weatherTest.nominal.arff"))
  val x1 = weatherTest.toArray(Array(new Array[Double](weatherTest.size())))
  val y1 = weatherTest.toArray(new Array[Int](weatherTest.size()))

  val error = x1.zip(y1).count(a => dTree.predict(a._1) != a._2)
  val decisions = x1.map{
    dTree.predict(_) match{
      case 0 => "play"
      case 1 => "not playable weather"
    }
  }.toList
  println("error is "+error)
  println(decisions)
}
