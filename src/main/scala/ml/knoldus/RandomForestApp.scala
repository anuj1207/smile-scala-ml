package ml.knoldus

import java.io.FileInputStream

import smile.classification._
import smile.data.parser.ArffParser

object RandomForestApp extends App{
  val arffParser = new ArffParser()
  arffParser.setResponseIndex(4)
  val weather = arffParser.parse(new FileInputStream("src/main/resources/weather.nominal.arff"))
  println("file loaded++++"+weather.attributes().toString)
  val x = weather.toArray(Array(new Array[Double](weather.size())))
  val y = weather.toArray(new Array[Int](weather.size()))

//  val forest = new AdaBoost(new Array[Attribute](2), x, y, 200)
  val rf = randomForest(x, y, weather.attributes(), 200, 4)
  private val trees = rf.getTrees
  trees.foreach(a => println(a.dot()))
  println(trees.size)

  println(s"OOB error = ${rf.error}")
  rf.predict(x(0))

}
