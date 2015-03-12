package opt.experiment

import opt.OptimizationAlgorithm
import opt.example.CitibikeProblem
import Util.averageOf
import Util.testNNAccuracy
import Util.time
import opt.OptimizationProblem

abstract class Experiment[Problem <: OptimizationProblem] {

  def main(args: Array[String]) = runTest

  def runTest: Unit

  def generateProblem: Problem with EvaluationCount

  private var problem: Problem with EvaluationCount = _
  private var algorithm: OptimizationAlgorithm = _

  def test(algorithmGenerators: Seq[Problem => OptimizationAlgorithm]): Unit = {
    val out = Console.out
    val err = Console.err

    // Neural Nets test
    for (algorithmGen <- algorithmGenerators) {
      var i = 0

      val avgMetrics = averageOf(10) {
        problem = generateProblem
        algorithm = algorithmGen(problem)
        if (i == 0) err.print(s"Running $algorithm on $problem")

        val metrics = generateTestMetrics(problem, algorithm)

        err.print(".")
        i += 1

        metrics
      }
      err.println
      err.flush

      out.println((Seq(problem, algorithm) ++ avgMetrics).mkString(", "))
    }
  }

  def generateTestMetrics(problem: Problem with EvaluationCount, algorithm: OptimizationAlgorithm): Seq[Double] = {
    val (_, timeMillis) = time(algorithm.train())
    val X = algorithm.getOptimal
    val fitness = problem.value(X)
    Seq(timeMillis, problem.evaluationCount, fitness)
  }
}
