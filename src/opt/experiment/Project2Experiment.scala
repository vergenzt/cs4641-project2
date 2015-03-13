package opt.experiment

import opt.OptimizationAlgorithm
import opt.RandomizedHillClimbing
import opt.SimulatedAnnealing
import opt.example.CitibikeProblem
import opt.ga.StandardGeneticAlgorithm
import opt.OptimizationProblem
import opt.prob.ProbabilisticOptimizationProblem
import opt.HillClimbingProblem
import opt.example.FlipFlopEvaluationFunction
import opt.example.FixedSubsetEvaluationFunction
import opt.prob.MIMIC
import opt.example.CountOnesEvaluationFunction

object TestRHC extends Project2Experiment { def runTest = testRHC }
object TestSA extends Project2Experiment { def runTest = testSA }
object TestGA extends Project2Experiment { def runTest = testGA }
object TestMIMIC extends Project2Experiment { def runTest = testMIMIC }
object TestAll extends Project2Experiment {
  def runTest = {
    testRHC
    testSA
    testGA
    testMIMIC
  }
}

abstract class Project2Experiment extends Experiment {
  import Util._

  val trainFile = "/citibike_discretized-8bins-eqwidth_normalized.train.csv"
  val testFile = "/citibike_discretized-8bins-eqwidth_normalized.test.csv"

  def problemGenerators: Seq[() => Problem with EvaluationCount] = Seq(
//    () => CitibikeProblem.apply(trainFile),
    () => new StandardGenericOptimizationProblem(new FlipFlopEvaluationFunction, 80) with EvaluationCount,
    () => new StandardGenericOptimizationProblem(new FixedSubsetEvaluationFunction(80, .5), 80) with EvaluationCount,
    () => new StandardGenericOptimizationProblem(new CountOnesEvaluationFunction, 80) with EvaluationCount
  )

  def testRHC = test(Seq[Problem => OptimizationAlgorithm](
    (problem) => new RandomizedHillClimbing(0, 5, problem),
    (problem) => new RandomizedHillClimbing(5, 5, problem),
    (problem) => new RandomizedHillClimbing(10, 5, problem),
    (problem) => new RandomizedHillClimbing(25, 5, problem),
    (problem) => new RandomizedHillClimbing(50, 5, problem),
    (problem) => new RandomizedHillClimbing(100, 5, problem),
    (problem) => new RandomizedHillClimbing(500, 5, problem)
  ))

  def testSA = test(Seq[Problem => OptimizationAlgorithm](
    (problem) => new SimulatedAnnealing(500, 10, 10, problem),
    (problem) => new SimulatedAnnealing(500, 25, 10, problem),
    (problem) => new SimulatedAnnealing(500, 50, 10, problem),
    (problem) => new SimulatedAnnealing(500, 100, 10, problem),
    (problem) => new SimulatedAnnealing(500, 500, 10, problem)
  ))

  def testGA = test(Seq[Problem => OptimizationAlgorithm](
    (problem) => new StandardGeneticAlgorithm(100, 50, 10, 10, problem),
    (problem) => new StandardGeneticAlgorithm(100, 50, 10, 25, problem),
    (problem) => new StandardGeneticAlgorithm(100, 50, 10, 50, problem),
    (problem) => new StandardGeneticAlgorithm(100, 50, 10, 100, problem),
    (problem) => new StandardGeneticAlgorithm(100, 50, 10, 500, problem)
  ))

  def testMIMIC = test(Seq[Problem => OptimizationAlgorithm](
    (problem) => new MIMIC(100, 50, 10, problem),
    (problem) => new MIMIC(100, 50, 25, problem),
    (problem) => new MIMIC(100, 50, 50, problem),
    (problem) => new MIMIC(100, 50, 100, problem),
    (problem) => new MIMIC(100, 50, 500, problem)
  ))

  override def generateTestMetrics(problem: Problem with EvaluationCount, algorithm: OptimizationAlgorithm) = {
    problem match {
      case citibikeProblem: CitibikeProblem => super.generateTestMetrics(problem, algorithm) ++
          Seq(trainFile, testFile).map(f => testNNAccuracy(citibikeProblem.network, f))
      case _ => super.generateTestMetrics(problem, algorithm)
    }
  }
}
