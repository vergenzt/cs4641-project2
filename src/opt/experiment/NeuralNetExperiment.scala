package opt.experiment

import scala.collection.mutable
import func.nn.NeuralNetwork
import opt.OptimizationAlgorithm
import opt.RandomizedHillClimbing
import opt.SimulatedAnnealing
import opt.example.CitibikeProblem
import shared.reader.DataSetLabelBinarySeperator
import shared.tester.AccuracyTestMetric
import shared.tester.NeuralNetworkTester
import scala.util.Random
import opt.ga.StandardGeneticAlgorithm

object TestRHC extends NeuralNetExperiment { def runTest = testRHC }
object TestSA extends NeuralNetExperiment { def runTest = testSA }
object TestGA extends NeuralNetExperiment { def runTest = testGA }
object TestAll extends NeuralNetExperiment {
  def runTest = {
    testRHC
    testSA
    testGA
  }
}

abstract class NeuralNetExperiment {
  import Util._

  val trainFile = "/citibike_discretized-8bins-eqwidth_normalized.train.csv"
  val testFile = "/citibike_discretized-8bins-eqwidth_normalized.test.csv"

  def main(args: Array[String]) = runTest

  def runTest: Unit

  def testRHC = test(Seq[CitibikeProblem => OptimizationAlgorithm](
    (problem) => new RandomizedHillClimbing(0, 5, problem),
    (problem) => new RandomizedHillClimbing(5, 5, problem),
    (problem) => new RandomizedHillClimbing(10, 5, problem),
    (problem) => new RandomizedHillClimbing(25, 5, problem),
    (problem) => new RandomizedHillClimbing(50, 5, problem)
  ))

  def testSA = test(Seq[CitibikeProblem => OptimizationAlgorithm](
    (problem) => new SimulatedAnnealing(100, 10, problem),
    (problem) => new SimulatedAnnealing(100, 25, problem),
    (problem) => new SimulatedAnnealing(10, 100, problem)
  ))

  def testGA = test(Seq[CitibikeProblem => OptimizationAlgorithm](
    (problem) => new StandardGeneticAlgorithm(50, 20, 10, 200, problem),
    (problem) => new StandardGeneticAlgorithm(200, 20, 10, 200, problem),
    (problem) => new StandardGeneticAlgorithm(200, 40, 10, 200, problem)
  ))

  def test(algorithmGenerators: Seq[CitibikeProblem => OptimizationAlgorithm]): Unit = {
    val out = Console.out
    val err = Console.err

    // Neural Nets test
    for (algorithmGen <- algorithmGenerators) {
      var problem: CitibikeProblem = null
      var algorithm: OptimizationAlgorithm = null
      var i = 0

      val Seq(avgTimeMillis, avgEvaluationCount, avgFitness, avgTrainAccuracy, avgTestAccuracy) = averageOf(10) {
        problem = CitibikeProblem(trainFile)
        algorithm = algorithmGen(problem)
        if (i == 0) err.print(s"Running $algorithm on $problem")

        val (_, timeMillis) = time(algorithm.train())
        val X = algorithm.getOptimal
        val fitness = problem.value(X)
        val trainAccuracy = testNNAccuracy(problem.network, trainFile)
        val testAccuracy = testNNAccuracy(problem.network, testFile)

        err.print(".")
        i += 1

        Seq(timeMillis, problem.evaluationCount, fitness, trainAccuracy, testAccuracy)
      }
      err.println
      err.flush

      out.println(s"$problem, $algorithm, $avgTimeMillis, $avgEvaluationCount, $avgFitness, $avgTrainAccuracy, $avgTestAccuracy")
    }
  }
}
