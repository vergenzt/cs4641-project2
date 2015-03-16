package opt.experiment

import Util.averageOf
import Util.time
import opt.HillClimbingProblem
import opt.OptimizationAlgorithm
import opt.OptimizationProblem
import opt.example.CitibikeProblem
import opt.ga.GeneticAlgorithmProblem
import opt.prob.MIMIC
import opt.prob.ProbabilisticOptimizationProblem
import scala.util.Try

abstract class Experiment {
  import Util._
  type Problem = OptimizationProblem
    with HillClimbingProblem
    with ProbabilisticOptimizationProblem
    with GeneticAlgorithmProblem

  def main(args: Array[String]) = runTest

  // abstract members
  def runTest: Unit
  def problemGenerators: Seq[() => Problem with EvaluationCount]

  // everything else...
  private var problem: Problem with EvaluationCount = _
  private var algorithm: OptimizationAlgorithm = _

  def test(algorithmGenerators: Seq[Problem => OptimizationAlgorithm]): Unit = {
    val out = Console.out
    val err = Console.err

    // Neural Nets test
    for {
      problemGen <- problemGenerators
      algorithmGen <- algorithmGenerators
      if (Try(algorithmGen(problemGen())).isSuccess)
    } {
      var i = 0

      val avgMetrics = averageOf(10) {
        problem = problemGen()
        algorithm = algorithmGen(problem)

        if (i == 0) err.print(s"Running $algorithm on $problem")

        val metrics = generateTestMetrics(problem, algorithm)

        err.print(".")
        i += 1

        metrics
      }
      err.println
      err.flush

      out.println((Seq(problem, algorithm).map("\""+_+"\"") ++ avgMetrics).mkString(", "))
    }
  }

  def generateTestMetrics(problem: Problem with EvaluationCount, algorithm: OptimizationAlgorithm): Seq[Double] = {
    val (_, timeMillis) = time(algorithm.train())
    val X = algorithm.getOptimal
    val fitness = problem.value(X)
    Seq(timeMillis, problem.evaluationCount, fitness)
  }
}
