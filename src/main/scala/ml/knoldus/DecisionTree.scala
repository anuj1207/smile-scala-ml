package ml.knoldus

import java.io.FileInputStream

//import smile.read
import smile.data.parser.ArffParser
import smile.classification.cart

object DecisionTree extends App{

  val arffParser = new ArffParser()
  arffParser.setResponseIndex(4)
  val weather = arffParser.parse(new FileInputStream("src/main/resources/weather.nominal.arff"))
  val x = weather.toArray(Array(new Array[Double](weather.size())))
  val y = weather.toArray(new Array[Int](weather.size()))

  val dTree = cart(x, y, 200, weather.attributes())
  val error = x.zipWithIndex.count(a => dTree.predict(a._1) != y(a._2))
  println("error is "+error)
}
