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
    (problem) => new RandomizedHillClimbing(50, 5, problem),
    (problem) => new RandomizedHillClimbing(100, 5, problem)
  ))

  def testSA = test(Seq[CitibikeProblem => OptimizationAlgorithm](
    (problem) => new SimulatedAnnealing(500, 10, 10, problem),
    (problem) => new SimulatedAnnealing(500, 25, 10, problem),
    (problem) => new SimulatedAnnealing(500, 50, 10, problem),
    (problem) => new SimulatedAnnealing(500, 100, 10, problem)
  ))

  def testGA = test(Seq[CitibikeProblem => OptimizationAlgorithm](
    (problem) => new StandardGeneticAlgorithm(100, 50, 10, 10, problem),
    (problem) => new StandardGeneticAlgorithm(100, 50, 10, 25, problem),
    (problem) => new StandardGeneticAlgorithm(100, 50, 10, 50, problem),
    (problem) => new StandardGeneticAlgorithm(100, 50, 10, 100, problem)
  ))

  override def generateTestMetrics(problem: CitibikeProblem with EvaluationCount, algorithm: OptimizationAlgorithm) = {
    super.generateTestMetrics(problem, algorithm) ++
      Seq(trainFile, testFile).map(f => testNNAccuracy(problem.network, f))
  }
}
