package opt.experiment

import opt.ga.GeneticAlgorithmProblem
import opt.prob.ProbabilisticOptimizationProblem
import opt.HillClimbingProblem
import opt.OptimizationProblem
import opt.ga.MutationFunction
import opt.NeighborFunction
import opt.ga.SingleCrossOver
import opt.DiscreteChangeOneNeighbor
import opt.ga.DiscreteChangeOneMutation
import opt.ga.CrossoverFunction
import opt.EvaluationFunction
import opt.GenericOptimizationProblem
import dist.Distribution
import dist.DiscreteDependencyTree
import dist.DiscreteUniformDistribution

class StandardGenericOptimizationProblem(val evalFn: EvaluationFunction, val N: Int)
  extends GenericOptimizationProblem(evalFn, new DiscreteUniformDistribution(Array.fill(N)(2)))
  with HillClimbingProblem
  with GeneticAlgorithmProblem
  with ProbabilisticOptimizationProblem {

  val ranges = Array.fill(N)(2)

  val nf: NeighborFunction  = new DiscreteChangeOneNeighbor(ranges);
  val mf: MutationFunction = new DiscreteChangeOneMutation(ranges);
  val cf: CrossoverFunction = new SingleCrossOver();
  val df: Distribution = new DiscreteDependencyTree(.1, ranges);

  // Members declared in opt.ga.GeneticAlgorithmProblem
  def mate(x$1: shared.Instance,x$2: shared.Instance): shared.Instance = cf.mate(x$1, x$2)
  def mutate(x$1: shared.Instance): Unit = mf.mutate(x$1)

  // Members declared in opt.HillClimbingProblem
  def neighbor(x$1: shared.Instance): shared.Instance = nf.neighbor(x$1)

  // Members declared in opt.prob.ProbabilisticOptimizationProblem
  def getDistribution(): dist.Distribution = df
}
