package opt.experiment

import opt.OptimizationAlgorithm
import opt.RandomizedHillClimbing
import opt.SimulatedAnnealing
import opt.example.CitibikeProblem
import opt.ga.StandardGeneticAlgorithm
import opt.GenericOptimizationProblem
import opt.example.FlipFlopEvaluationFunction
import dist.DiscreteUniformDistribution

object Test_FF_RHC extends FlipFlopExperiment { def runTest = testRHC }
object Test_FF_SA extends FlipFlopExperiment { def runTest = testSA }
object Test_FF_GA extends FlipFlopExperiment { def runTest = testGA }
object Test_FF_MMC extends FlipFlopExperiment { def runTest = testMMC }
object Test_FF_All extends FlipFlopExperiment {
  def runTest = {
    testRHC
    testSA
    testGA
    testMMC
  }
}

abstract class FlipFlopExperiment extends Experiment[StandardGenericOptimizationProblem] {

  def generateProblem = {
    new StandardGenericOptimizationProblem(new FlipFlopEvaluationFunction, 80)
      with EvaluationCount
  }

  def testRHC = test(Seq[StandardGenericOptimizationProblem => OptimizationAlgorithm](
    (problem) => new RandomizedHillClimbing(0, 5, problem),
    (problem) => new RandomizedHillClimbing(5, 5, problem),
    (problem) => new RandomizedHillClimbing(10, 5, problem),
    (problem) => new RandomizedHillClimbing(25, 5, problem),
    (problem) => new RandomizedHillClimbing(50, 5, problem)
  ))

  def testSA = test(Seq[StandardGenericOptimizationProblem => OptimizationAlgorithm](
    (problem) => new SimulatedAnnealing(100, 10, 10, problem),
    (problem) => new SimulatedAnnealing(100, 25, 10, problem),
    (problem) => new SimulatedAnnealing(10, 100, 10, problem)
  ))

  def testGA = test(Seq[StandardGenericOptimizationProblem => OptimizationAlgorithm](
    (problem) => new StandardGeneticAlgorithm(50, 20, 10, 200, problem),
    (problem) => new StandardGeneticAlgorithm(200, 20, 10, 200, problem),
    (problem) => new StandardGeneticAlgorithm(200, 40, 10, 200, problem)
  ))

  def testMMC = test(Seq[StandardGenericOptimizationProblem => OptimizationAlgorithm](
    (problem) => new StandardGeneticAlgorithm(50, 20, 10, 200, problem),
    (problem) => new StandardGeneticAlgorithm(200, 20, 10, 200, problem),
    (problem) => new StandardGeneticAlgorithm(200, 40, 10, 200, problem)
  ))
}
