package opt.experiment

import func.nn.NeuralNetwork
import shared.tester.NeuralNetworkTester
import shared.reader.DataSetLabelBinarySeperator
import opt.example.CitibikeProblem
import shared.tester.AccuracyTestMetric

object Util {
  def testNNAccuracy(network: NeuralNetwork, testFilename: String) = {
    val accuracyMetric = new AccuracyTestMetric
    val tester = new NeuralNetworkTester(network, accuracyMetric)
    val testData = CitibikeProblem.getDataSet(testFilename, 29)
    DataSetLabelBinarySeperator.seperateLabels(testData)
    tester.test(testData.getInstances)
    accuracyMetric.getPctCorrect
  }

  def time[A](block: => A): (A, Long) = {
    val start = System.nanoTime
    val ret = block
    val end = System.nanoTime
    (ret, (end - start)/1000)
  }

  def averageOf(n: Int)(block: => Seq[Double]) = {
    Seq.fill(n)(block).transpose.map(_.sum / n)
  }

  def maxOf(n: Int)(block: => Double) = {
    Seq.fill(n)(block).max
  }

  def progressBar(i: Int, n: Int): String = {
    "[" + "="*(i-1) + (if (i>0) ">" else "") + " "*(n-i) + "]"
  }

  def printTabular(data: Seq[Seq[Any]]) = {
    val strings = data.map(_.map(_.toString))
    val widths = strings.transpose.map(_.map(_.length).max)
    strings.foreach(row => println(row.zip(widths)
        .map({case (value,width) => " "*(width-value.length) + value})
        .mkString(" | ")))
  }
}
