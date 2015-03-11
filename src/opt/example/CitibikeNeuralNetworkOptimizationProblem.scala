package opt.example

import func.nn.NeuralNetwork
import func.nn.activation.LogisticSigmoid
import func.nn.feedfwd.FeedForwardNeuralNetworkFactory
import shared.DataSet
import shared.ErrorMeasure
import shared.SumOfSquaresError
import shared.reader.ArffDataSetReader
import shared.reader.DataSetLabelBinarySeperator

class CitibikeProblem(
  val trainFilename: String,
  val testFilename: String,
  val data: DataSet,
  val network: NeuralNetwork,
  val error: ErrorMeasure
) extends NeuralNetworkOptimizationProblem(data, network, error) {

  override def toString = "Citibike"
}

object CitibikeProblem {
  def apply(trainFilename: String, testFilename: String) = {
    val data = new ArffDataSetReader(trainFilename).read(29)
    DataSetLabelBinarySeperator.seperateLabels(data)
    val numInput = 29
    val numOutput = 8
    val numHidden = (numInput + numOutput) / 2
    val factory = new FeedForwardNeuralNetworkFactory()
    val network = factory.createClassificationNetwork(Array(numInput, numHidden, numOutput), new LogisticSigmoid)
    val error = new SumOfSquaresError
    new CitibikeProblem(trainFilename, testFilename, data, network, error)
  }
}
