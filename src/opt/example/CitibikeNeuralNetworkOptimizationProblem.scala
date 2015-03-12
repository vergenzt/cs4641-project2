package opt.example

import scala.io.Source
import scala.util.Try
import func.nn.NeuralNetwork
import func.nn.activation.LogisticSigmoid
import func.nn.feedfwd.FeedForwardNeuralNetworkFactory
import shared.DataSet
import shared.ErrorMeasure
import shared.Instance
import shared.SumOfSquaresError
import shared.reader.DataSetLabelBinarySeperator
import util.linalg.DenseVector
import opt.experiment.EvaluationCount

class CitibikeProblem(
  val trainFilename: String,
  val data: DataSet,
  val network: NeuralNetwork,
  val error: ErrorMeasure
) extends NeuralNetworkOptimizationProblem(data, network, error) with EvaluationCount {

  override def toString = "Citibike"
}

object CitibikeProblem {
  def apply(trainFilename: String) = {
    val data = getDataSet(trainFilename, 29)
    DataSetLabelBinarySeperator.seperateLabels(data)
    val numInput = 29
    val numOutput = 8
    val numHidden = (numInput + numOutput) / 2
    val factory = new FeedForwardNeuralNetworkFactory()
    val network = factory.createClassificationNetwork(Array(numInput, numHidden, numOutput), new LogisticSigmoid)
    val error = new SumOfSquaresError
    new CitibikeProblem(trainFilename, data, network, error)
  }

  def getDataSet(filename: String, labelIndex: Int = -1) =
    new DataSet((
      for {
        line <- Source.fromURL(getClass.getResource(filename)).getLines
        parts = line.split(',').map(_.trim)
        values = parts.map(s =>
          Try(s.toDouble).getOrElse(0.0)
        )
        (xValues, yValues) = values.splitAt(if (labelIndex < 0) values.size - 1 else labelIndex + 1)
        (xVector, yVector) = (new DenseVector(xValues), new DenseVector(yValues))
      } yield {
        new Instance(xVector, new Instance(yVector))
      }
    ).toArray)
}
