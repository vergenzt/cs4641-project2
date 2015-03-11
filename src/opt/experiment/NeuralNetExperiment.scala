package opt.experiment

import scala.annotation.migration
import opt.HillClimbingProblem
import opt.OptimizationProblem
import opt.RandomizedHillClimbing
import opt.SimulatedAnnealing
import opt.example.CitibikeProblem
import opt.ga.GeneticAlgorithmProblem
import opt.ga.StandardGeneticAlgorithm
import shared.tester.NeuralNetworkTester
import sun.swing.AccumulativeRunnable
import shared.tester.AccuracyTestMetric
import shared.reader.ArffDataSetReader
import shared.reader.DataSetLabelBinarySeperator

object NeuralNetExperiment {

  val trainFile = "/home/tim/Schoolwork/CS4641/project2/ABAGAIL/data/citibike_discretized-8bins-eqwidth_normalized.train.arff"
  val testFile = "/home/tim/Schoolwork/CS4641/project2/ABAGAIL/data/citibike_discretized-8bins-eqwidth_normalized.test.arff"

  def testCases = Seq(
    (() => CitibikeProblem.apply(trainFile, testFile),
      Seq(
        (problem: OptimizationProblem) => new RandomizedHillClimbing(problem.asInstanceOf[HillClimbingProblem]),
        (problem: OptimizationProblem) => new SimulatedAnnealing(1000, 1000, problem.asInstanceOf[HillClimbingProblem]),
        (problem: OptimizationProblem) => new StandardGeneticAlgorithm(200, 100, 10, 100, problem.asInstanceOf[GeneticAlgorithmProblem])
      )
    )
  )

  def main(args: Array[String]) = {

    println("Beginning simulations...")

    val results = for {
      (problemGenerator, algorithmGenerators) <- testCases
      algorithmGenerator <- algorithmGenerators
    } yield {
      val problem = problemGenerator()
      val algorithm = algorithmGenerator(problem)
      print(s"\rRunning $algorithm on $problem: ${progressBar(0, 10)}")

      val Seq(avgBestFitness, avgAccuracy, avgTimeMillis) = averageOf(10) { i =>
        val problem = problemGenerator()
        val algorithm = algorithmGenerator(problem)
        print(s"\rRunning $algorithm on $problem: ${progressBar(i+1, 10)}")

        val (bestX, timeMillis) = time {
          algorithm.train()
          algorithm.getOptimal
        }
        val bestFitness = problem.value(bestX)

        // test on NN test data
        val accuracy = if (problem.isInstanceOf[CitibikeProblem]) {
          val accuracyMetric = new AccuracyTestMetric
          val tester = new NeuralNetworkTester(problem.network, accuracyMetric)
          val testData = new ArffDataSetReader(testFile).read(29)
          DataSetLabelBinarySeperator.seperateLabels(testData)
          tester.test(testData.getInstances)
          accuracyMetric.getPctCorrect
        } else -1

        Seq(bestFitness, accuracy, timeMillis)
      }

      Seq(problem.toString, algorithm.toString, avgBestFitness, avgAccuracy, avgTimeMillis)
    }

    println
    val header = Seq("Problem", "Algorithm", "AvgBestFitness", "AvgAccuracy", "AvgTimeMillis")
    printTabular(header +: results.sortBy(-1*_(2).asInstanceOf[Double]))

  }

  private def time[A](block: => A): (A, Long) = {
    val start = System.nanoTime
    val ret = block
    val end = System.nanoTime
    (ret, (end - start)/1000)
  }

  private def averageOf(n: Int)(block: Int => Seq[Double]) = {
    Seq.tabulate(n)(block).transpose.map(_.sum / n)
  }

  private def progressBar(i: Int, n: Int): String = {
    "[" + "="*(i-1) + (if (i>0) ">" else "") + " "*(n-i) + "]"
  }

  private def printTabular(data: Seq[Seq[Any]]) = {
    val strings = data.map(_.map(_.toString))
    val widths = strings.transpose.map(_.map(_.length).max)
    strings.foreach(row => println(row.zip(widths)
        .map({case (value,width) => " "*(width-value.length) + value})
        .mkString(" | ")))
  }
}
