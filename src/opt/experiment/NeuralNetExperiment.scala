package opt.experiment

import opt.OptimizationAlgorithm
import opt.RandomizedHillClimbing
import opt.SimulatedAnnealing
import opt.example.CitibikeProblem
import opt.ga.StandardGeneticAlgorithm

object Test_NN_RHC extends NeuralNetExperiment { def runTest = testRHC }
object Test_NN_SA extends NeuralNetExperiment { def runTest = testSA }
object Test_NN_GA extends NeuralNetExperiment { def runTest = testGA }
object Test_NN_All extends NeuralNetExperiment {
  def runTest = {
    testRHC
    testSA
    testGA
  }
}

abstract class NeuralNetExperiment extends Experiment[CitibikeProblem] {
  import Util._

  val trainFile = "/citibike_discretized-8bins-eqwidth_normalized.train.csv"
  val testFile = "/citibike_discretized-8bins-eqwidth_normalized.test.csv"

  def generateProblem = CitibikeProblem.apply(trainFile)

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

  override def generateTestMetrics(problem: CitibikeProblem with EvaluationCount, algorithm: OptimizationAlgorithm) = {
    super.generateTestMetrics(problem, algorithm) ++
      Seq(trainFile, testFile).map(f => testNNAccuracy(problem.network, f))
  }
}
