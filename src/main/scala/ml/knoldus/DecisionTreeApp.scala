package ml.knoldus

import smile.classification.DecisionTree.SplitRule.ENTROPY
import smile.classification.cart
import smile.read
import smile.data

object DecisionTreeApp extends App{

  val weather = read.arff("src/main/resources/weather.nominal.arff", 4)/*returns object of attributeDataSet*/

  val trainingInstances = weather.toArray(Array(new Array[Double](weather.size())))/*finds Array training instances
  //ex Array(Array(sunny,hot,high,false,no)....) but in double type like sunny is replaced by 0 its index
  //if data is like @attribute outlook{sunny, overcast, rainy}*/

  val responseVariables = weather.toArray(new Array[Int](weather.size()))/*returns responses for a single array
  *ex Array(yes, no....) in Int type like yes is replced by 0 its index
  *if data is like @attribute play{yes, no}*/

  private val maxNodes = 200
  private val attributes = weather.attributes()
  private val splitRule = ENTROPY
  val dTree = cart(trainingInstances, responseVariables, maxNodes, attributes, splitRule)
  /*cart(classification and regression tree) that takes test data and returns a decision tree
  //takes training instances (Array of Array of double) 
  //and response instances(Array of Int) 
  //and then Array of all attributes that are there in data file using @attribute annotation and the index is set on their order
  //last is the operation we are performing to get information gain for each attribute on that basis we'll be getting out dTree*/

  val tree = dTree.dot()/*it return graphic representation in Graphviz dot format */
  println(tree)// here we are printing the decision tree
  println("\n\ncopy and paste the above on this link to print tree >>>>> http://viz-js.com/\n\n")

  /**training is completed now
   Testing will start*/

  /*Now again we are parsing a Test file which contains 12 samples with error*/
  val weatherTest = read.arff("src/main/resources/weatherTest.nominal.arff", 4)

  /*again testInstances is the Array of testInstances where each value is an Array of Double*/
  val testInstances = weatherTest.toArray(Array(new Array[Double](weatherTest.size())))
  /*this testResponseInstances is the Array of response values that are in int type*/
  val testResponseVariables = weatherTest.toArray(new Array[Int](weatherTest.size()))

  /*here we are predicting the responses using predict method of decisionTree
    * and checking these out puts with the responses provided in test file to check how many error are there in test data file*/
  val error = testInstances.zip(testResponseVariables).count {
    case (testInstance, response) => dTree.predict(testInstance) != response
  }

  println("Number of errors in test data is "+error)

  /*In the following we are using the predict method to find out whether the day is suitable to play or not
    * here we are using the test instances provided in the test data*/
  val decisions = testInstances.map{
    dTree.predict(_) match{
      case 0 => "play"
      case 1 => "not playable weather"
    }
  }.toList

  /*printing the list of decisions*/
  println(decisions)
}

